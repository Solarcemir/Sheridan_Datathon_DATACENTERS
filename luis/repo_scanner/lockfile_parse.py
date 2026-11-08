"""Turn a lockfile's raw text into (ecosystem, name, version) triples. MVP scope: npm and PyPI -- the two
ecosystems OSV.dev covers well and that cover most student/hackathon-scale repos. We only ever read the
lockfile, never the source: no package here touches the rest of a repo even if more of it is uploaded.

Deliberately NOT a general SBOM tool: real ones (Syft, osv-scanner itself) handle many more ecosystems and
edge cases. This is the minimum that makes the demo honest and correct for its two ecosystems.
"""
import json
import re

NAME_VERSION_PIP = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.\-]*)\s*==\s*([0-9][0-9A-Za-z.\-+_!]*)")


def parse_npm_lock(text):
    """package-lock.json, lockfileVersion 1-3. Returns a de-duplicated {name: version} dict."""
    data = json.loads(text)
    deps = {}
    packages = data.get("packages")
    if packages:   # v2/v3: keys are "node_modules/<name>" (or scoped "node_modules/@scope/name"); "" is the root
        for path, info in packages.items():
            if not path or "version" not in info:
                continue
            name = path.rsplit("node_modules/", 1)[-1]
            deps[name] = info["version"]
    else:   # v1: nested "dependencies"
        def walk(node):
            for name, info in (node or {}).items():
                if "version" in info:
                    deps[name] = info["version"]
                if "dependencies" in info:
                    walk(info["dependencies"])
        walk(data.get("dependencies", {}))
    return deps


def parse_requirements_txt(text):
    """requirements.txt with pinned versions (name==1.2.3). Unpinned/editable/VCS lines are skipped --
    they have no single resolvable version for OSV to match against."""
    deps = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = NAME_VERSION_PIP.match(line)
        if m:
            deps[m.group(1)] = m.group(2)
    return deps


def detect_and_parse(filename, text):
    """-> (ecosystem, {name: version}). Raises ValueError for anything not recognized (MVP scope)."""
    name = (filename or "").lower()
    if name.endswith("package-lock.json") or (text.lstrip().startswith("{") and '"lockfileVersion"' in text):
        return "npm", parse_npm_lock(text)
    if name.endswith("requirements.txt") or NAME_VERSION_PIP.search(text or ""):
        deps = parse_requirements_txt(text)
        if deps:
            return "PyPI", deps
    raise ValueError("Unrecognized lockfile. This build reads package-lock.json (npm) and requirements.txt "
                      "(PyPI, pinned versions only) -- paste one of those, or its contents.")
