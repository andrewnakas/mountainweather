"""Offline checks of watchdog.py's slot logic, with a faked GitHub API.

Nothing here touches GitHub. Run after changing watchdog.py:
    python3 .github/watchdog/test_watchdog.py
"""
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

spec = importlib.util.spec_from_file_location("watchdog", Path(__file__).with_name("watchdog.py"))
wd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wd)

T = lambda hm: datetime(2026, 11, 23, int(hm[:2]), int(hm[3:]), tzinfo=timezone.utc)
S = lambda hm: f"2026-11-23T{hm}:00Z"

RUNS, JOBS, CALLS, POSTS = {}, {}, [], []


def fake_api(method, path, body=None):
    CALLS.append((method, path))
    if method == "POST":
        POSTS.append(path)
        return None
    if "/jobs" in path:
        rid = int(path.split("/runs/")[1].split("/")[0])
        if rid == 999:
            raise OSError("boom")
        return {"jobs": JOBS.get(rid, [])}
    return {"workflow_runs": RUNS["list"]}


wd.api = fake_api
FAIL = []


def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAIL.append(name)


def run(rid, created, updated, status, conclusion=None, title="wind velocity"):
    return {"id": rid, "created_at": S(created), "updated_at": S(updated), "status": status,
            "conclusion": conclusion, "display_title": title}


slot = T("07:15")
# The 23rd's case: made 07:07 for the 06:15 slot, picked up 08:08, done 08:09.
r = run(1, "07:07", "08:09", "completed", "success")
JOBS[1] = [{"started_at": S("08:08"), "runner_name": "GitHub Actions 1"}]
ok("late run picked up after the slot covers it", wd.covers("o/r", r, slot))
JOBS[2] = [{"started_at": S("07:10"), "runner_name": "GitHub Actions 2"}]
ok("run picked up before the slot does not", not wd.covers("o/r", run(2, "07:07", "07:20", "completed", "success"), slot))
CALLS.clear()
ok("run finished before the slot does not, without a jobs call",
   not wd.covers("o/r", run(3, "06:20", "06:30", "completed", "success"), slot) and not CALLS)
ok("queued run made earlier covers", wd.covers("o/r", run(4, "07:07", "07:07", "queued"), slot))
ok("pending run made earlier covers", wd.covers("o/r", run(5, "06:40", "06:40", "pending"), slot))
ok("run made after the slot covers", wd.covers("o/r", run(6, "07:20", "07:30", "completed", "failure"), slot))
ok("cancelled run never covers", not wd.covers("o/r", run(7, "07:20", "07:30", "completed", "cancelled"), slot))
JOBS[8] = [{"started_at": S("07:50"), "runner_name": "GitHub Actions 8"}]
ok("in-progress run picked up after the slot covers", wd.covers("o/r", run(8, "07:07", "07:50", "in_progress"), slot))
JOBS[9] = [{"started_at": S("07:50"), "runner_name": None, "conclusion": "skipped"}]
ok("run whose jobs were all skipped does not", not wd.covers("o/r", run(9, "07:07", "07:50", "completed", "success"), slot))
ok("unreadable jobs mean dispatch", not wd.covers("o/r", run(999, "07:07", "07:50", "completed", "success"), slot))

# check(): the 08:09 dispatch would not happen now, and a real gap still dispatches.
task = {"workflow": "wind.yml", "title": "wind velocity", "grace": 30, "minute": 15, "hours": None,
        "inputs": {"task": "velocity"}, "label": "wind velocity"}
RUNS["list"] = [run(1, "07:07", "08:09", "completed", "success")]
POSTS.clear()
got = wd.check("o/r", "main", [task], T("08:10"), dry_run=False)
ok("check: no duplicate behind the late run", got == [] and not POSTS)
RUNS["list"] = [run(2, "07:07", "07:20", "completed", "success")]
POSTS.clear()
got = wd.check("o/r", "main", [task], T("08:10"), dry_run=False)
ok("check: a slot with no run after it dispatches", got == ["wind velocity"] and POSTS)
RUNS["list"] = [run(10, "05:00", "05:00", "queued")]
POSTS.clear()
got = wd.check("o/r", "main", [task], T("08:10"), dry_run=False)
ok("check: a run stuck queued >120 min is cancelled and replaced",
   got == ["wind velocity"] and any(p.endswith("/cancel") for p in POSTS) and any(p.endswith("/dispatches") for p in POSTS))

print("ALL PASS" if not FAIL else f"FAILURES {FAIL}")
sys.exit(1 if FAIL else 0)
