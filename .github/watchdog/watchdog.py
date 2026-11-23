#!/usr/bin/env python3
"""
Start this repository's scheduled runs when GitHub's scheduler drops them.

GitHub runs `schedule:` crons on a best-effort basis, and in November 2026 it
dropped most of the crons in these repositories for hours at a time (about
01:20Z-06:00Z on the 20th, 02:20Z-11:10Z on the 22nd) while its status page
stayed green. The map layers these pipelines feed went stale until someone
dispatched the runs by hand.

This loop is that someone. Every few minutes it asks, for each task in
tasks.json: has a run started since the task's last cron slot? Once the slot
is more than `grace` minutes old with nothing started, it dispatches the task.
The crons stay where they are and normally get there first; the watchdog only
fills in what they miss, so on a good day it dispatches nothing.

It also clears the other failure seen that month: a run that sits "queued"
for hours with no runner ever assigned. Cancelling it and dispatching a
fresh one got it going within minutes every time.

A run that is only waiting its turn is left alone, however late it is. On the
22nd runners took about an hour to arrive for every run on the account, and
cancelling at 60 minutes sent each run to the back of the queue just before
its turn came. Such a run also counts as the current slot's run: it fetches
the latest data when it starts, so a second one behind it would only repeat
the work.

workflow_dispatch is the one event the repository's own GITHUB_TOKEN is
allowed to trigger, so no personal token is involved. watchdog.yml runs this
for most of the six-hour job limit and then dispatches its own successor.

Check once from a laptop, changing nothing:
    GITHUB_TOKEN=$(gh auth token) GITHUB_REPOSITORY=owner/repo \\
        python3 .github/watchdog/watchdog.py --once --dry-run
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

API = os.environ.get("GITHUB_API_URL", "https://api.github.com")
HERE = Path(__file__).resolve().parent
WORKFLOWS = HERE.parent / "workflows"

# Queued this long with no runner means stuck, not waiting for a turn. Waits of
# an hour were normal on the 22nd; the stuck runs seen that month sat for hours.
STUCK_AFTER = timedelta(minutes=120)

# Made but not started yet.
NOT_STARTED = {"queued", "pending", "waiting", "requested"}


def api(method, path, body=None):
    req = urllib.request.Request(
        f"{API}{path}",
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else None


def parse_time(stamp):
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def parse_cron(expr):
    """(minute, hours) for the crons used here: a fixed minute, any hour field.

    hours is None for every hour. Day, month and weekday must be `*`; nothing
    in these repositories needs more, and refusing is better than guessing.
    """
    fields = expr.split()
    if len(fields) != 5 or fields[2:] != ["*", "*", "*"]:
        raise ValueError(f"unsupported cron {expr!r}")
    minute, hour = int(fields[0]), fields[1]
    if not 0 <= minute <= 59:
        raise ValueError(f"bad minute in {expr!r}")
    if hour == "*":
        return minute, None
    if hour.startswith("*/"):
        return minute, list(range(0, 24, int(hour[2:])))
    hours = [int(h) for h in hour.split(",")]
    if not all(0 <= h <= 23 for h in hours):
        raise ValueError(f"bad hour in {expr!r}")
    return minute, hours


def last_due_slot(task, now):
    """The most recent cron slot whose grace period has run out by `now`."""
    latest = (now - timedelta(minutes=task["grace"])).replace(second=0, microsecond=0)
    slot = latest.replace(minute=task["minute"])
    if slot > latest:
        slot -= timedelta(hours=1)
    while task["hours"] is not None and slot.hour not in task["hours"]:
        slot -= timedelta(hours=1)
    return slot


def load_tasks():
    tasks = json.loads((HERE / "tasks.json").read_text())["tasks"]
    for task in tasks:
        task["minute"], task["hours"] = parse_cron(task["cron"])
        task.setdefault("inputs", {})
        task["label"] = task.get("title") or task["workflow"].removesuffix(".yml")
        # tasks.json mirrors the workflow's own `schedule:`. Say so loudly if
        # the two have drifted, rather than watching a slot nobody runs.
        source = WORKFLOWS / task["workflow"]
        if source.exists() and task["cron"] not in source.read_text():
            print(f"::warning::{task['label']}: cron {task['cron']!r} is not in "
                  f"{task['workflow']}; update tasks.json to match", flush=True)
    return tasks


def check(repo, ref, tasks, now, dry_run):
    """One pass over every task. Returns the labels it dispatched."""
    listed = {}
    started = []
    for task in tasks:
        try:
            slot = last_due_slot(task, now)
            if task["workflow"] not in listed:
                listed[task["workflow"]] = api(
                    "GET", f"/repos/{repo}/actions/workflows/{task['workflow']}/runs?per_page=100"
                )["workflow_runs"]
            mine = [
                r for r in listed[task["workflow"]]
                if "title" not in task or r["display_title"] == task["title"]
            ]
            # Stuck from any slot, not just the latest. A run left over from
            # an earlier slot holds the workflow's concurrency group, so the
            # run for the current slot waits "pending" behind it for as long
            # as it sits there. Timed from updated_at, which is when a run
            # waiting on its group became queued for a runner.
            stuck = [
                r for r in mine
                if r["status"] == "queued" and now - parse_time(r["updated_at"]) > STUCK_AFTER
            ]
            for run in stuck:
                print(f"{now:%H:%MZ} {task['label']}: run {run['id']} has been queued "
                      f"since {parse_time(run['updated_at']):%H:%MZ} with no runner, "
                      f"{'would cancel' if dry_run else 'cancelling'}", flush=True)
                if not dry_run:
                    api("POST", f"/repos/{repo}/actions/runs/{run['id']}/cancel")
            # A run made for an earlier slot that has not started yet covers
            # this slot too, since it fetches the latest data when it starts.
            runs = [
                r for r in mine
                if (parse_time(r["created_at"]) >= slot or r["status"] in NOT_STARTED)
                and r.get("conclusion") != "cancelled"
                and r not in stuck
            ]
            if runs:
                continue
            print(f"{now:%H:%MZ} {task['label']}: nothing running since the {slot:%H:%MZ} slot, "
                  f"{'would dispatch' if dry_run else 'dispatching'}", flush=True)
            if not dry_run:
                api("POST", f"/repos/{repo}/actions/workflows/{task['workflow']}/dispatches",
                    {"ref": ref, "inputs": task["inputs"]})
            started.append(task["label"])
        except Exception as exc:  # one bad call must not stop the other tasks
            print(f"{now:%H:%MZ} {task['label']}: check failed: {exc!r}", flush=True)
    return started


def main():
    ap = argparse.ArgumentParser(description="Dispatch scheduled runs GitHub dropped.")
    ap.add_argument("--once", action="store_true", help="check once and exit")
    ap.add_argument("--dry-run", action="store_true", help="report only, dispatch nothing")
    ap.add_argument("--minutes", type=float, default=340, help="how long to keep watching")
    ap.add_argument("--every", type=int, default=300, help="seconds between checks")
    args = ap.parse_args()

    repo = os.environ["GITHUB_REPOSITORY"]
    tasks = load_tasks()
    ref = api("GET", f"/repos/{repo}")["default_branch"]
    print(f"watching {len(tasks)} tasks in {repo}@{ref}: "
          + ", ".join(t["label"] for t in tasks), flush=True)

    stop = time.monotonic() + args.minutes * 60
    started = []
    while True:
        now = datetime.now(timezone.utc)
        found = check(repo, ref, tasks, now, args.dry_run)
        if not found:
            print(f"{now:%H:%MZ} all on schedule", flush=True)
        started += [f"{now:%H:%MZ} {label}" for label in found]
        if args.once or time.monotonic() + args.every > stop:
            break
        time.sleep(args.every)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(f"Dispatched {len(started)} run(s) the schedule missed.\n\n")
            fh.writelines(f"- {line}\n" for line in started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
