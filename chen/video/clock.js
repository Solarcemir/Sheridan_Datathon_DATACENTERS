/* Virtual clock for frame-exact capture (injected by render.py into every frame, the scanner iframe included).
   Page time only moves when the renderer calls __advance(ms): timers, requestAnimationFrame, performance.now/Date.now
   and every CSS animation/transition are stepped by hand, so a slow screenshot never makes the animation skip.
   Network requests (OSV, GitHub) still run in real time; they just resolve between frames. */
(function () {
  var now = 0, T0 = Date.now(), seq = 1, timers = new Map(), raf = new Map(), anims = new Map();
  performance.now = function () { return now; };
  var RealDate = Date; Date.now = function () { return T0 + now; };
  window.setTimeout = function (fn, ms) { var a = [].slice.call(arguments, 2), id = seq++; timers.set(id, { at: now + Math.max(0, +ms || 0), fn: fn, a: a }); return id; };
  window.setInterval = function (fn, ms) { var a = [].slice.call(arguments, 2), id = seq++; timers.set(id, { at: now + Math.max(1, +ms || 0), fn: fn, a: a, every: Math.max(1, +ms || 0) }); return id; };
  window.clearTimeout = window.clearInterval = function (id) { timers.delete(id); };
  window.requestAnimationFrame = function (fn) { var id = seq++; raf.set(id, fn); return id; };
  window.cancelAnimationFrame = function (id) { raf.delete(id); };
  function runTimers(until) {
    for (var guard = 0; guard < 10000; guard++) {
      var best = null, bid = 0;
      timers.forEach(function (t, id) { if (t.at <= until && (!best || t.at < best.at)) { best = t; bid = id; } });
      if (!best) break;
      now = Math.max(now, best.at);
      if (best.every) best.at += best.every; else timers.delete(bid);
      try { typeof best.fn === "function" ? best.fn.apply(window, best.a) : eval(best.fn); } catch (e) { console.error(e); }
    }
  }
  function stepAnimations(dt) {
    var list = document.getAnimations ? document.getAnimations() : [];
    list.forEach(function (a) {
      if (!anims.has(a)) { a.pause(); a.currentTime = 0; anims.set(a, 0); return; }   // new this frame: starts now
      var v = anims.get(a) + dt; anims.set(a, v);
      var eff = a.effect && a.effect.getComputedTiming ? a.effect.getComputedTiming() : null;
      if (eff && isFinite(eff.endTime) && v >= eff.endTime) { anims.delete(a); try { a.finish(); } catch (e) {} }
      else a.currentTime = v;
    });
  }
  window.__advance = function (ms) {
    var target = now + ms;
    runTimers(target); now = target;
    var cbs = raf; raf = new Map(); cbs.forEach(function (fn) { try { fn(now); } catch (e) { console.error(e); } });
    runTimers(now);
    stepAnimations(ms);
    var f = document.querySelector("iframe");
    if (f && f.contentWindow && f.contentWindow.__advance) try { f.contentWindow.__advance(ms); } catch (e) {}
  };
  window.__now = function () { return now; };
})();
