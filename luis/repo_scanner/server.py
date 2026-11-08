"""The whole product behind one command: `python -E -P luis/repo_scanner/server.py`.

Serves the scan API (POST /api/scan) and the frontend's static files (luis/frontend/) from the same
process on the same origin, so the page can `fetch("/api/scan")` with no CORS setup needed. Open
http://127.0.0.1:8900/repo-scan.html after starting it.

Stdlib only (http.server) -- no new runtime dependency to install just to try this.
"""
import json
import mimetypes
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scan  # noqa: E402
import github_scan  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(HERE, "..", "frontend"))
CHEN_FRONTEND_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "chen", "frontend"))
PORT = int(os.environ.get("PORT", 8900))
MAX_BODY = 8 * 1024 * 1024   # 8 MB: generous for any real lockfile, small enough to not be an open door


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print("[server] " + (fmt % args))

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/api/scan", "/api/scan-github"):
            return self._json(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > MAX_BODY:
            return self._json(400, {"error": "missing or too-large request body"})
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except Exception:
            return self._json(400, {"error": "expected JSON"})
        try:
            if path == "/api/scan-github":
                url = payload.get("url", "")
                if not url.strip():
                    return self._json(400, {"error": "missing \"url\""})
                result = github_scan.scan_github_repo(url, scan.run_scan, scan.build_upgrade_plan)
            else:
                filename, content = payload.get("filename", ""), payload.get("content", "")
                if not content.strip():
                    return self._json(400, {"error": "empty lockfile content"})
                result = scan.run_scan(filename, content)
        except ValueError as e:   # unrecognized lockfile, bad URL, clone failure -- a user-facing message, not a crash
            return self._json(422, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            return self._json(500, {"error": f"scan failed: {e}"})
        return self._json(200, result)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            path = "/repo-scan.html"
        root = FRONTEND_DIR
        if path.startswith("/chen/"):   # Chen's model presentation, served read-only from his folder (never modified)
            root, path = CHEN_FRONTEND_DIR, path[len("/chen"):]
        local = os.path.normpath(os.path.join(root, path.lstrip("/")))
        if not local.startswith(root) or not os.path.isfile(local):
            return self._json(404, {"error": "not found"})
        ctype = mimetypes.guess_type(local)[0] or "application/octet-stream"
        data = open(local, "rb").read()
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text/") or "json" in ctype else ""))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    print("[server] loading the trained model + local NVD/KEV snapshot (first request will be a little slower)...")
    scan._ensure_models()
    print(f"[server] exploit indexes loaded: {scan.ls.warm_indexes()}")
    print(f"[server] ready. Open http://127.0.0.1:{PORT}/repo-scan.html", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
