"""GitHub URL input: `git clone --depth 1` a public repo into a throwaway temp dir, find every
package-lock.json / requirements.txt in it, scan each through scan.run_scan(), merge into one ranked
result, then delete the clone. The clone never outlives this one call -- we only ever needed the
lockfiles, and deleting the rest is the whole point of the privacy promise in repo-scan.html.

Scope, deliberately: github.com only (no arbitrary git:// / ssh:// targets -- this stays a repo
scanner, not a general-purpose git-clone-and-fetch service), public repos only (no credentials are
ever passed to git), depth-1 clone (small, fast, no history needed).
"""
import os
import re
import shutil
import subprocess
import tempfile
import time

GITHUB_RE = re.compile(r"^(?:https?://github\.com/|github\.com/)?([\w.-]+)/([\w.-]+?)(?:\.git)?/?(?:/tree/[\w./-]+)?$")
LOCKFILE_NAMES = {"package-lock.json", "requirements.txt"}
SKIP_DIRS = {".git", "node_modules", "vendor", ".venv", "venv", "dist", "build", "__pycache__"}
MAX_LOCKFILES = 8
CLONE_TIMEOUT = 50


def normalize(url_or_shorthand):
    """"IBM/repo", "github.com/IBM/repo" or a full https URL -> (clone_url, "IBM/repo"). Raises ValueError
    for anything that isn't a plain github.com org/repo reference."""
    s = url_or_shorthand.strip()
    m = GITHUB_RE.match(s)
    if not m:
        raise ValueError("Expected a GitHub repo: \"org/repo\", \"github.com/org/repo\" or a full "
                          "https://github.com/org/repo URL. Other hosts and git/ssh URLs aren't supported.")
    org, repo = m.group(1), m.group(2)
    return f"https://github.com/{org}/{repo}.git", f"{org}/{repo}"


def find_lockfiles(root):
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            if fn in LOCKFILE_NAMES:
                hits.append(os.path.join(dirpath, fn))
        if len(hits) >= MAX_LOCKFILES:
            break
    return hits[:MAX_LOCKFILES]


def clone(clone_url, dest):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")   # never let git hang prompting for credentials
    r = subprocess.run(["git", "clone", "--depth", "1", "--quiet", "--single-branch", clone_url, dest],
                        capture_output=True, text=True, timeout=CLONE_TIMEOUT, env=env)
    if r.returncode != 0:
        msg = (r.stderr or "").strip().splitlines()[-1] if r.stderr else "unknown error"
        raise ValueError(f"Could not clone it (private repo, wrong name, or no network): {msg[:200]}")


def scan_github_repo(url_or_shorthand, run_scan, build_upgrade_plan):
    """run_scan / build_upgrade_plan: from scan.py, passed in rather than imported to avoid a circular import."""
    clone_url, display = normalize(url_or_shorthand)
    tmp = tempfile.mkdtemp(prefix="repo_scan_gh_")
    t0 = time.time()
    try:
        clone(clone_url, tmp)
        lockfiles = find_lockfiles(tmp)
        if not lockfiles:
            raise ValueError(f"No package-lock.json or requirements.txt found in {display} "
                              "(default branch, depth 1 -- a lockfile only on another branch won't be seen).")
        per_file, all_rows, all_kev = [], [], []
        for path in lockfiles:
            rel = os.path.relpath(path, tmp).replace("\\", "/")
            content = open(path, encoding="utf-8", errors="ignore").read()
            try:
                result = run_scan(os.path.basename(path), content)
            except ValueError:
                continue   # a malformed/empty lockfile in the repo -- skip it, don't fail the whole scan
            per_file.append({"path": rel, "summary": result["summary"]})
            for row in result["rows"]:
                row["source_lockfile"] = rel
                all_rows.append(row)
            for row in result["already_in_kev"]:
                row["source_lockfile"] = rel
                all_kev.append(row)
        all_rows.sort(key=lambda r: -r["score"])
        for i, r in enumerate(all_rows, 1):
            r["rank"] = i
        return {
            "repo": display, "lockfiles_found": [f["path"] for f in per_file], "per_file": per_file,
            "elapsed_s": round(time.time() - t0, 1),
            "summary": {
                "dependencies": sum(f["summary"]["dependencies"] for f in per_file),
                "cves_found": sum(f["summary"]["cves_found"] for f in per_file),
                "already_in_kev": len(all_kev), "scored": len(all_rows),
                "demoted": sum(r["demoted"] for r in all_rows),
                "rejected_ids_skipped": sum(f["summary"].get("rejected_ids_skipped", 0) for f in per_file),
                "lockfiles": len(per_file),
            },
            "already_in_kev": all_kev, "rows": all_rows,
            "upgrade_plan": build_upgrade_plan(all_rows, all_kev),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)   # the clone is gone the moment we're done with it
