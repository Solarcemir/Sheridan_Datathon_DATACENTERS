"""Fake demo backend -- NOT a real app, exists only to give the repo scanner's demo_repo a believable
reason to pin the old package versions in requirements.txt. Do not run this; it is deliberately insecure
in the exact ways its pinned dependencies are, which is the whole point of the demo.
"""
from flask import Flask, request, render_template_string
import yaml
import requests
from PIL import Image
import io

app = Flask(__name__)


@app.route("/parse-config", methods=["POST"])
def parse_config():
    # yaml.load() without Loader=SafeLoader -- arbitrary code execution on PyYAML < 5.3.1 (CVE-2020-1747 etc.)
    config = yaml.load(request.data, Loader=yaml.Loader)
    return {"config": config}


@app.route("/thumbnail", methods=["POST"])
def thumbnail():
    img = Image.open(io.BytesIO(request.data))
    img.thumbnail((128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), 200, {"Content-Type": "image/png"}


@app.route("/fetch")
def fetch():
    url = request.args.get("url", "")
    r = requests.get(url, timeout=5)
    return {"status": r.status_code, "length": len(r.content)}


@app.route("/greet")
def greet():
    name = request.args.get("name", "world")
    return render_template_string("Hello, {{ name }}!", name=name)


if __name__ == "__main__":
    app.run(debug=True)
