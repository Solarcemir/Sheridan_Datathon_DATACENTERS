# demo_repo — a fake, deliberately vulnerable sample app

Not a real product. Exists only so the repo scanner has something realistic to point at for a demo —
a tiny two-service app (a Python backend, a Node frontend), each with a lockfile pinned to real,
old, known-vulnerable dependency versions. **Don't run or deploy this.** The code is just enough to
look like a real app; the versions in the lockfiles are the actual point.

## Try it

```
python -E -P luis/repo_scanner/scan.py luis/repo_scanner/demo_repo/backend/requirements.txt
python -E -P luis/repo_scanner/scan.py luis/repo_scanner/demo_repo/frontend/package-lock.json
```
Or start `luis/repo_scanner/server.py` and paste either file's contents into `repo-scan.html`.

## What's pinned, and why

| File | Package | Version | Why it's here |
|---|---|---|---|
| `backend/requirements.txt` | PyYAML | 5.1 | `app.py`'s `/parse-config` route calls `yaml.load()` on raw request bytes — PyYAML < 5.3.1 lets that run arbitrary code (CVE-2020-1747, CVE-2020-14343). |
| | Pillow | 8.2.0 | Used to thumbnail an uploaded image in `/thumbnail`. Several real Pillow CVEs land in this version range. |
| | requests / urllib3 | 2.19.0 / 1.24.1 | Used by `/fetch` to pull a remote URL. urllib3 1.24.1 has a real CRLF-injection CVE (CVE-2019-11236). |
| | Flask / Jinja2 | 0.12 / 2.10 | The web framework and its templating engine, both old enough to have their own CVE history. |
| `frontend/package-lock.json` | lodash | 4.17.4 | `server.js`'s `/greet` route runs `_.template` on user input — lodash's template-injection CVEs (e.g. CVE-2021-23337) are exactly this pattern. |
| | minimist | 0.0.8 | Used to parse a CLI flag in `server.js`. minimist < 1.2.2 has a real prototype-pollution CVE (CVE-2020-7598). |
| | node-fetch | 2.6.0 | Used by `/fetch` on the Node side, mirroring the Python backend's route. |

Expect the scanner to rank the PyYAML and lodash findings near the top (public exploit code + remote
code execution / injection framing in the description) and EPSS-only entries like old `requests`
pinned versions lower — that contrast is the point of the demo.
