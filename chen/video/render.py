"""Render the ~2 min demo video, frame by frame, from our own pages (no screen recording).

  1. galaxy.html  - 3D opening: all 72,670 scoreboard CVEs, real final-model scores, collapsing to our 500
  2. the deck     - frontend/breach_radar.html?video=1 under a virtual clock (clock.js): overview with data
                    flowing along the wires, the tour (slides 2-7), then a live scan in the demo, then an end card
  3. ffmpeg       - both segments at 30 fps, 1 s crossfade, H.264 1080p

Run from anywhere while chen/ is served on 127.0.0.1:8093:
  uv run --with playwright --with imageio-ffmpeg python chen/video/render.py OUT_DIR [--only galaxy|deck|join]
Timings live in GALAXY_S and DECK (seconds); VOICEOVER.md is written against them.
"""
import asyncio, os, subprocess, sys, shutil, time
from playwright.async_api import async_playwright

BASE = "http://127.0.0.1:8093"
FPS, W, H = 30, 1920, 1080
GALAXY_S = 22.5
SPARSE = int(sys.argv[sys.argv.index("--sparse") + 1]) if "--sparse" in sys.argv else 1   # test runs: keep every Nth frame
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "IBM/MAX-Object-Detector"

# deck timeline: (second, action). Slide indices are PATH positions: 0 problem, 1 data, 2 features, 3 voters,
# 4 cleanup, 5 proof, 6 top500; "demo" = the locked demo. Slide 1 is skipped: the galaxy already told it.
DECK = [
    (0.0, "overview"),
    (7.0, ("go", 1)),
    (15.0, ("go", 2)),
    (24.0, ("go", 3)),
    (32.0, ("go", 4)),
    (39.0, ("go", 5)),
    (47.0, ("go", 6)),
    (54.0, "demo"),
    (57.0, "scroll_scan"),
    (59.0, "pick_repo"),
    (60.5, "scan"),
    (76.0, "scroll_results"),
    (84.0, "end_card"),
]
DECK_S = 90.0

END_CARD = """(() => { const d = document.createElement('div'); d.id = 'endcard';
  d.innerHTML = '<div style="font:800 140px/1 Archivo,sans-serif;font-stretch:80%;letter-spacing:-.01em">KEV<span style="color:var(--alert)">exploit</span></div>' +
    '<div style="margin-top:22px;font:500 34px IBM Plex Sans,sans-serif;color:var(--muted)">Patch <b style="color:var(--alert)">these 500</b> first.</div>' +
    '<div style="margin-top:46px;font:500 20px IBM Plex Mono,monospace;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)">Team DATACENTERS · Sheridan Datathon 2026 · IBM track</div>';
  d.style.cssText = 'position:fixed;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;background:var(--bg);color:var(--ink);z-index:50;opacity:0;transition:opacity 1.2s';
  document.body.appendChild(d); requestAnimationFrame(() => requestAnimationFrame(() => { d.style.opacity = 1; })); })()"""


def ff():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def encode(frames, out):
    subprocess.run([ff(), "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(frames, "%05d.jpg"),
                    "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", out], check=True)


async def galaxy(b, out):
    pg = await b.new_page(viewport={"width": W, "height": H})
    await pg.goto(BASE + "/video/galaxy.html"); await pg.evaluate("window.ready"); await pg.evaluate("document.fonts.ready")
    n = int(GALAXY_S * FPS)
    for i in range(n):
        await pg.evaluate(f"renderAt({i / FPS})")
        await pg.screenshot(path=os.path.join(out, f"{i:05d}.jpg"), type="jpeg", quality=93)
        if i % 150 == 0: print(f"galaxy {i}/{n}", flush=True)
    await pg.close()


async def deck(b, out):
    ctx = await b.new_context(viewport={"width": W, "height": H}, color_scheme="dark")
    await ctx.add_init_script(path=os.path.join(HERE, "clock.js"))
    pg = await ctx.new_page()
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    await pg.goto(BASE + "/frontend/breach_radar.html?video=1#problem")
    await pg.wait_for_function("window.__deck && document.querySelector('#demo').contentWindow.__advance", polling=200)   # rAF polling would wait forever: clock.js owns rAF
    await pg.evaluate("document.fonts.ready")
    for _ in range(90): await pg.evaluate("__advance(1000/30)")   # settle boot (3 s of page time, not filmed)
    await pg.evaluate("__deck.go(-1)")
    for _ in range(75): await pg.evaluate("__advance(1000/30)")
    fr = pg.frame_locator("#demo")
    scroll = None   # (frame js target, from, to, t0, t1)
    n, todo = int(DECK_S * FPS), list(DECK)
    for i in range(n):
        t = i / FPS
        while todo and todo[0][0] <= t + 1e-9:
            _, a = todo.pop(0)
            if a == "overview": await pg.evaluate("__deck.go(-1)")
            elif a == "demo": await pg.evaluate("__deck.go(__deck.DEMO)")
            elif isinstance(a, tuple) and a[0] == "go": await pg.evaluate(f"__deck.go({a[1]})")
            elif a == "scroll_scan":
                y = await fr.locator("#scan").evaluate("e => e.getBoundingClientRect().top + scrollY - 30")
                scroll = (0, y, t, t + 1.6)
            elif a == "pick_repo": await fr.locator(f'[data-gh="{REPO}"]').click()
            elif a == "scan": await fr.locator("#scan-btn").click()
            elif a == "scroll_results":
                y0 = await fr.locator("html").evaluate("e => scrollY")
                y = await fr.locator("#results").evaluate("e => e.getBoundingClientRect().top + scrollY - 20")
                scroll = (y0, y, t, t + 2.2)
            elif a == "end_card": await pg.evaluate(END_CARD)
        if scroll:
            y0, y1, t0, t1 = scroll; k = min(1, max(0, (t - t0) / (t1 - t0))); e = k * k * (3 - 2 * k)
            await fr.locator("html").evaluate(f"e => window.scrollTo(0, {y0 + (y1 - y0) * e})")
            if k >= 1: scroll = None
        await pg.evaluate("__advance(1000/30)")
        if i % SPARSE == 0: await pg.screenshot(path=os.path.join(out, f"{i:05d}.jpg"), type="jpeg", quality=93)
        if i % 150 == 0: print(f"deck {i}/{n} t={t:.1f}", flush=True)
    if errs: print("page errors:", errs)
    await ctx.close()


async def main():
    out = sys.argv[1]; only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    os.makedirs(out, exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        for name, fn in (("galaxy", galaxy), ("deck", deck)):
            if only and only != name: continue
            d = os.path.join(out, name); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
            t0 = time.time(); await fn(b, d)
            if SPARSE == 1: encode(d, os.path.join(out, name + ".mp4"))
            print(name, "done in", round(time.time() - t0), "s", flush=True)
        await b.close()
    if SPARSE == 1 and (not only or only == "join"):
        subprocess.run([ff(), "-y", "-loglevel", "error", "-i", os.path.join(out, "galaxy.mp4"), "-i", os.path.join(out, "deck.mp4"),
                        "-filter_complex", f"[0][1]xfade=transition=fade:duration=1:offset={GALAXY_S - 1}[v]", "-map", "[v]",
                        "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", os.path.join(out, "kevexploit_demo.mp4")], check=True)
        print("wrote", os.path.join(out, "kevexploit_demo.mp4"))

asyncio.run(main())
