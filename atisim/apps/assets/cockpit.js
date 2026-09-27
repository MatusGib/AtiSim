/* The live cockpit (atisim/apps/fly.py), in the browser.

   One flight per cockpit page. The page starts a flight (POST /api/flight),
   then, while flying, sends the held keys about 25 times a second and gets one
   frame of numbers back: the server runs whole 50 Hz physics steps up to the
   wall clock, exactly as scripts/fly.py does. Everything that moves is drawn
   here on one canvas, from those numbers:

     - the primary flight display: airspeed tape, attitude, altitude tape,
       vertical speed, heading tape, with the start condition as magenta bugs;
     - the flight-test instruments beside it: load factor with its peaks, angle
       of attack against the model's linear band, wind, and the gust rate, which
       is labelled SIM TRUTH because no instrument can sense it;
     - a strip of the load factor over the last 40 s.

   Drawing runs at the display's rate and interpolates between the two frames
   around (now - DELAY_MS), so the horizon moves smoothly between polls.

   The arrow keys are a centre stick, as on the matplotlib panel: up is stick
   forward, so up pitches the nose DOWN. The keycaps on the card say so. */
(function () {
  "use strict";

  var KEYMAP = {
    ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right",
    ",": ",", ".": ".", "<": ",", ">": ".", "-": "-", "_": "-", "=": "=", "+": "=",
    "[": "[", "]": "]", "{": "[", "}": "]"
  };
  var POLL_MS = 40;
  var DELAY_MS = 70;
  var STRIP_S = 40;
  var FONT = '"B612", system-ui, sans-serif';
  var HAND = '"Permanent Marker", "Segoe Print", cursive';
  var C = {
    glare: "#0a0b0d", panel: "#15181c", tape: "#1d2126", edge: "#3a4148",
    text: "#f2f3f5", dim: "#a7aeb6", sky: "#2a6fbd", ground: "#7c5231",
    horizon: "#ffffff", symbol: "#ffd24a", magenta: "#ff5ad9", green: "#3ddc84",
    amber: "#ffb020", red: "#ff5a4e", trace: "#7fd1ff", grid: "#262c33",
    ink: "#16181b", ink2: "#474c53", paper: "#fbfbf9", rule: "#d5d9de",
    stamp: "#1b4d9b", pencil: "#2b2724", apWash: "#e6edf7"
  };

  var S = null;  // the session of the cockpit page on screen

  // ---- lifecycle: the page is a Dash page, so it comes and goes ------------
  function watch() {
    var root = document.getElementById("cockpit");
    if (root && (!S || S.root !== root)) {
      if (S) { stop(); }
      begin(root);
    } else if (!root && S) {
      stop();
    }
  }
  new MutationObserver(watch).observe(document.documentElement, {childList: true, subtree: true});
  document.addEventListener("DOMContentLoaded", watch);

  function begin(root) {
    var canvas = root.querySelector("#fly-pfd");
    S = {
      root: root, tp: root.dataset.tp, n: root.dataset.tpN,
      canvas: canvas, ctx: canvas.getContext("2d"),
      overlay: root.querySelector("#fly-overlay"),
      held: new Set(), presses: {a: 0, t: 0},
      frames: [], trace: [], peak: null, info: null, id: null,
      state: "loading", alive: true, timer: 0, raf: 0, inflight: null, sideAt: 0
    };
    var s = S;
    s.onKeyDown = function (e) { onKeyDown(s, e); };
    s.onKeyUp = function (e) { var k = KEYMAP[e.key]; if (k) { s.held.delete(k); } };
    s.onBlur = function () { s.held.clear(); };
    s.onHide = function () { if (document.hidden && s.state === "flying") { pause(s); } };
    s.onClick = function (e) { onClick(s, e); };
    window.addEventListener("keydown", s.onKeyDown);
    window.addEventListener("keyup", s.onKeyUp);
    window.addEventListener("blur", s.onBlur);
    document.addEventListener("visibilitychange", s.onHide);
    root.addEventListener("click", s.onClick);
    s.resize = new ResizeObserver(function () { size(s); });
    s.resize.observe(canvas.parentElement);
    size(s);
    var fonts = document.fonts ? Promise.all([
      document.fonts.load("700 16px B612"), document.fonts.load("16px B612"),
      document.fonts.load("16px 'B612 Mono'"), document.fonts.load("20px 'Permanent Marker'")
    ]).catch(function () {}) : Promise.resolve();
    fonts.then(function () { if (s.alive) { create(s); } });
    s.raf = requestAnimationFrame(function tick() {
      if (!s.alive) { return; }
      draw(s);
      s.raf = requestAnimationFrame(tick);
    });
  }

  function stop() {
    var s = S;
    S = null;
    s.alive = false;
    clearTimeout(s.timer);
    cancelAnimationFrame(s.raf);
    window.removeEventListener("keydown", s.onKeyDown);
    window.removeEventListener("keyup", s.onKeyUp);
    window.removeEventListener("blur", s.onBlur);
    document.removeEventListener("visibilitychange", s.onHide);
    s.resize.disconnect();
    if (s.id && s.state === "flying") {
      fetch("/api/flight/" + s.id + "/pause", {method: "POST", keepalive: true}).catch(function () {});
    }
  }

  function size(s) {
    var box = s.canvas.parentElement.getBoundingClientRect();
    var dpr = window.devicePixelRatio || 1;
    s.W = Math.max(300, box.width);
    s.H = Math.max(300, box.height);
    s.canvas.width = Math.round(s.W * dpr);
    s.canvas.height = Math.round(s.H * dpr);
    s.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // ---- the server ------------------------------------------------------------
  function post(url, body) {
    return fetch(url, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body || {})
    }).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (data) {
        if (!r.ok) {
          var err = new Error(data.error || ("the server answered " + r.status));
          err.status = r.status;
          throw err;
        }
        return data;
      });
    });
  }

  function create(s) {
    setState(s, "loading");
    note(s, '<h2><span class="fly-spinner" aria-hidden="true"></span>Preparing the aircraft</h2>' +
      "<p>Trimming the 747 for level flight at its cruise condition and compiling " +
      "the flight model. The first flight after a start takes a few seconds; the " +
      "next ones do not.</p>");
    s.frames = []; s.trace = []; s.peak = null; s.presses = {a: 0, t: 0}; s.id = null;
    post("/api/flight", {tp: s.tp}).then(function (r) {
      if (!s.alive) { return; }
      s.id = r.id;
      s.info = r.info;
      receive(s, r.frame);
      setState(s, "ready");
      var f = r.frame, tgt = r.info.targets;
      note(s, "<h2>Ready at " + fmt(tgt.airspeed, 0) + " m/s, " + group(tgt.altitude) + " m</h2>" +
        "<p>The 747 is trimmed for level flight, heading north. " + escapeHtml(ahead(s, f, true)) + "</p>" +
        "<p>Arrow keys fly it; the up arrow is stick forward and puts the nose down. " +
        "Press <b>A</b> at any time to let the autopilot hold height, speed and heading.</p>" +
        '<div class="fly-note-row"><button type="button" class="fly-btn fly-btn-primary" data-act="start">' +
        "Start the flight</button></div><p style=\"margin:10px 0 0;font-size:13.5px\">or press Space</p>");
      focusIn(s, '[data-act="start"]');
    }, function (err) { fail(s, err); });
  }

  function loop(s) {
    if (!s.alive || s.state !== "flying") { return; }
    var sent = performance.now();
    s.inflight = post("/api/flight/" + s.id + "/poll", {held: Array.from(s.held), presses: s.presses});
    s.inflight.then(function (f) {
      s.inflight = null;
      if (!s.alive) { return; }
      receive(s, f);
      if (s.state === "flying") {
        s.timer = setTimeout(function () { loop(s); }, Math.max(0, POLL_MS - (performance.now() - sent)));
      }
    }, function (err) { s.inflight = null; if (s.alive) { fail(s, err); } });
  }

  function receive(s, f) {
    s.frames.push({at: performance.now(), f: f});
    if (s.frames.length > 12) { s.frames.shift(); }
    if (!s.peak) { s.peak = {max: f.nz, min: f.nz, tmax: f.t, tmin: f.t}; }
    if (f.nz > s.peak.max) { s.peak.max = f.nz; s.peak.tmax = f.t; }
    if (f.nz < s.peak.min) { s.peak.min = f.nz; s.peak.tmin = f.t; }
    var last = s.trace[s.trace.length - 1];
    if (!last || f.t > last.t) { s.trace.push({t: f.t, nz: f.nz}); }
    while (s.trace.length && s.trace[0].t < f.t - STRIP_S - 1) { s.trace.shift(); }
    var now = performance.now();
    if (now - s.sideAt > 150) { s.sideAt = now; side(s, f); }
  }

  // ---- states -----------------------------------------------------------------
  function setState(s, state) {
    s.state = state;
    var q = function (sel) { return s.root.querySelector(sel); };
    var flyable = state === "ready" || state === "flying" || state === "paused";
    q("#fly-pause").disabled = !flyable;
    q("#fly-ap").disabled = !(state === "flying" || state === "paused");
    q("#fly-end").disabled = !(state === "flying" || state === "paused");
    q("#fly-pause").classList.toggle("is-flying", state === "flying");
    q("#fly-pause-word").textContent = state === "flying" ? "Pause" : state === "paused" ? "Resume" : "Start";
  }

  function startFlying(s) {
    if (s.state !== "ready" && s.state !== "paused") { return; }
    clear(s);
    setState(s, "flying");
    loop(s);
  }

  function pause(s) {
    if (s.state !== "flying") { return; }
    setState(s, "paused");
    clearTimeout(s.timer);
    // After any poll already on its way, so the server's clock stops last.
    (s.inflight || Promise.resolve()).then(function () {
      return post("/api/flight/" + s.id + "/pause");
    }).catch(function () {});
    note(s, "<h2>Paused</h2><p>The flight is frozen at t = " + fmt(latest(s).t, 1) +
      " s. Nothing moves until you resume.</p>" +
      '<div class="fly-note-row"><button type="button" class="fly-btn fly-btn-primary" data-act="resume">Resume</button>' +
      '<button type="button" class="fly-btn" data-act="end">End flight and read the recorder</button></div>', true);
    focusIn(s, '[data-act="resume"]');
  }

  function end(s) {
    if (s.state !== "flying" && s.state !== "paused") { return; }
    clearTimeout(s.timer);
    setState(s, "reading");
    note(s, '<h2><span class="fly-spinner" aria-hidden="true"></span>Reading the recorder</h2>' +
      "<p>Computing the load factor and angle of attack at every physics step of your flight.</p>", true);
    (s.inflight || Promise.resolve()).then(function () {
      return post("/api/flight/" + s.id + "/summary");
    }).then(function (sum) {
      if (!s.alive) { return; }
      setState(s, "debrief");
      debrief(s, sum);
    }, function (err) { if (s.alive) { fail(s, err); } });
  }

  function autopilot(s) {
    if (s.state !== "flying") { return; }
    s.presses.a += 1;
  }

  function fail(s, err) {
    clearTimeout(s.timer);
    setState(s, "error");
    var gone = err && err.status === 404;
    note(s, "<h2>" + (gone ? "This flight has ended" : "The flight stopped") + "</h2><p>" +
      escapeHtml(gone ? "The application restarted, or newer flights replaced this one."
                      : String(err && err.message || err)) + "</p>" +
      '<div class="fly-note-row"><button type="button" class="fly-btn fly-btn-primary" data-act="again">' +
      "Fly TP-" + escapeHtml(s.n) + " again</button>" +
      '<button type="button" class="fly-btn" data-act="card">Back to the test card</button></div>', true);
  }

  // ---- input ------------------------------------------------------------------
  function onKeyDown(s, e) {
    if (e.ctrlKey || e.metaKey || e.altKey) { return; }
    var tag = e.target && e.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") { return; }
    var k = KEYMAP[e.key];
    if (k) {
      e.preventDefault();
      s.held.add(k);
      return;
    }
    var key = (e.key || "").toLowerCase();
    if (key === "a" && !e.repeat) { autopilot(s); }
    else if (key === "t" && !e.repeat && s.state === "flying") { s.presses.t += 1; }
    else if (e.key === " " || e.key === "Spacebar") {
      if (tag === "BUTTON" || tag === "A") { return; }  // the focused control's own click
      e.preventDefault();
      if (e.repeat) { return; }
      if (s.state === "flying") { pause(s); } else { startFlying(s); }
    } else if (e.key === "Escape" && s.state === "flying") { pause(s); }
  }

  function onClick(s, e) {
    var el = e.target.closest("button, a[data-act]");
    if (!el) { return; }
    var act = el.dataset.act || {"fly-pause": "toggle", "fly-ap": "ap", "fly-end": "end"}[el.id];
    if (!act) { return; }
    e.preventDefault();
    if (act === "start" || act === "resume") { startFlying(s); }
    else if (act === "toggle") { if (s.state === "flying") { pause(s); } else { startFlying(s); } }
    else if (act === "ap") { autopilot(s); }
    else if (act === "end") { end(s); }
    else if (act === "again") { clear(s); create(s); }
    else if (act === "card") { navigate("/"); }
    else if (act === "workbench") { navigate("/start"); }
  }

  // A Dash page change without a reload: dcc.Location follows this event.
  function navigate(href) {
    window.history.pushState({}, "", href);
    window.dispatchEvent(new CustomEvent("_dashprivate_pushstate"));
    window.dispatchEvent(new PopStateEvent("popstate"));
  }

  // ---- the kneeboard's live fields ------------------------------------------------
  function side(s, f) {
    var aheadEl = s.root.querySelector("#fly-ahead");
    if (aheadEl) { aheadEl.textContent = ahead(s, f, false); }
    var peakEl = s.root.querySelector("#fly-peak");
    if (peakEl && s.peak && s.state !== "loading" && f.t > 0) {
      peakEl.innerHTML = '<span class="tc-pencil">' + signed(s.peak.max, 2) + " g</span>" +
        '<span class="tc-pencil-note" style="margin-left:10px">low ' + signed(s.peak.min, 2) + " g</span>";
    }
    var ap = s.root.querySelector("#fly-ap");
    if (ap) {
      var on = f.mode === "autopilot";
      ap.setAttribute("aria-pressed", on ? "true" : "false");
      s.root.querySelector("#fly-ap-word").textContent = on ? "Autopilot on" : "Autopilot off";
    }
  }

  function ahead(s, f, sentence) {
    var fld = f.field;
    if (!fld) { return sentence ? "The air is still: there is nothing ahead." : "Still air"; }
    var secs = fld.closing > 1 ? fld.distance / fld.closing : null;
    if (fld.bearing === null) {  // a line vortex: a north distance, no bearing
      var core = fld.label.replace(/^.* core /, "core ");
      if (fld.distance < 0) {
        return sentence ? "You are past the last core."
                        : "Past the last core. End the flight to read the recorder.";
      }
      var txt = cap(core) + " in " + km(fld.distance) + (secs ? ", " + fmt(secs, 0) + " s" : "");
      return sentence ? "The first vortex core is " + km(fld.distance) + " ahead" +
        (secs ? ", " + fmt(secs, 0) + " s at this speed." : ".") : txt;
    }
    if (fld.closing < -1) {
      return sentence ? "The updraft column is behind you."
                      : "Column behind you. End the flight to read the recorder.";
    }
    return sentence ? "The updraft column's centre is " + km(fld.distance) + " ahead" +
      (secs ? ", " + fmt(secs, 0) + " s at this speed." : ".")
      : "Column centre in " + km(fld.distance) + (secs ? ", " + fmt(secs, 0) + " s" : "");
  }

  // ---- overlay notes -------------------------------------------------------------
  function note(s, html, dim) {
    s.overlay.classList.toggle("is-dim", !!dim);
    s.overlay.innerHTML = '<div class="fly-note" role="status">' + html + "</div>";
  }
  function clear(s) { s.overlay.classList.remove("is-dim"); s.overlay.innerHTML = ""; }
  function focusIn(s, sel) {
    var el = s.overlay.querySelector(sel);
    if (el) { el.focus({preventScroll: true}); }
  }

  // ---- drawing ---------------------------------------------------------------------
  function latest(s) { return s.frames.length ? s.frames[s.frames.length - 1].f : null; }

  function sample(s) {
    var fr = s.frames;
    if (!fr.length) { return null; }
    if (s.state !== "flying" || fr.length === 1) { return fr[fr.length - 1].f; }
    var at = performance.now() - DELAY_MS;
    if (at <= fr[0].at) { return fr[0].f; }
    for (var i = fr.length - 1; i > 0; i--) {
      var a = fr[i - 1], b = fr[i];
      if (at >= a.at && at <= b.at) {
        var u = (at - a.at) / Math.max(1, b.at - a.at);
        return lerpFrame(a.f, b.f, u);
      }
    }
    return fr[fr.length - 1].f;
  }

  function lerpFrame(a, b, u) {
    var o = Object.assign({}, b);
    ["t", "airspeed", "altitude", "vs", "pitch", "bank", "alpha", "beta", "nz", "ny", "throttle"]
      .forEach(function (k) { o[k] = a[k] + (b[k] - a[k]) * u; });
    var dh = ((b.heading - a.heading + 540) % 360) - 180;
    o.heading = (a.heading + dh * u + 360) % 360;
    return o;
  }

  function layout(W, H) {
    var m = 16;
    var narrow = W < 700;
    var instW = narrow ? 0 : clamp(W * 0.2, 170, 232);
    var stripH = clamp(H * 0.16, 80, 132);
    var fmaH = 38;
    var hdgH = 46;
    var top = m + fmaH + 14;
    var bottom = H - m - stripH - 16 - hdgH - 10;
    var mainH = Math.max(120, bottom - top);
    var x1 = W - m - instW - (narrow ? 0 : 22);
    var tapeW = 84, altW = 100, vsiW = 40, gap = 12;
    var S0 = Math.min(mainH, x1 - m - tapeW - altW - vsiW - 3 * gap - 8);
    var Sz = Math.max(120, S0);
    var block = tapeW + gap + Sz + gap + altW + 8 + vsiW;
    var x0 = m + Math.max(0, (x1 - m - block) / 2);
    var y0 = top + (mainH - Sz) / 2;
    return {
      m: m, W: W, H: H, S: Sz, narrow: narrow,
      fma: {x: m, y: m, w: W - 2 * m, h: fmaH},
      spd: {x: x0, y: y0, w: tapeW, h: Sz},
      att: {x: x0 + tapeW + gap, y: y0, s: Sz},
      alt: {x: x0 + tapeW + gap + Sz + gap, y: y0, w: altW, h: Sz},
      vsi: {x: x0 + block - vsiW, y: y0 + Sz * 0.08, w: vsiW, h: Sz * 0.84},
      hdg: {x: x0 + tapeW + gap - 6, y: y0 + Sz + 12, w: Sz + 12, h: hdgH},
      inst: {x: W - m - instW, y: top, w: instW, h: y0 + Sz + 12 + hdgH - top},
      strip: {x: m, y: H - m - stripH, w: W - 2 * m, h: stripH}
    };
  }

  function draw(s) {
    var ctx = s.ctx, W = s.W, H = s.H;
    ctx.fillStyle = C.glare;
    ctx.fillRect(0, 0, W, H);
    var f = sample(s);
    var L = layout(W, H);
    if (!f || !s.info) {
      text(ctx, "PFD", L.att.x + L.S / 2, L.att.y + L.S / 2, 14, C.dim, "center", 700);
      frameBox(ctx, L.att.x, L.att.y, L.S, L.S, 10);
      return;
    }
    var lim = s.info.limits, tgt = s.info.targets;
    drawFma(ctx, L.fma, s, f);
    drawAttitude(ctx, L.att, f, lim);
    // Half-spans as the matplotlib panel's tapes: 40 m/s, and 300 m of altitude.
    drawTape(ctx, L.spd, f.airspeed, tgt.airspeed, L.S / 2 / 40, 5, 10, "left", "M/S", 0);
    drawTape(ctx, L.alt, f.altitude, tgt.altitude, L.S / 2 / 300, 20, 100, "right", "M", 0);
    drawVsi(ctx, L.vsi, f.vs, lim.vsi_span);
    drawHeading(ctx, L.hdg, f.heading, tgt.heading);
    if (!L.narrow) { drawInstruments(ctx, L.inst, s, f, lim); }
    drawStrip(ctx, L.strip, s, f, lim);
  }

  function drawFma(ctx, r, s, f) {
    var ap = f.mode === "autopilot";
    var x = r.x, h = r.h, y = r.y;
    x = annunciator(ctx, x, y, h, ap ? "AUTOPILOT" : "MANUAL", ap ? C.green : C.text, ap);
    if (r.w < 700) { annunciator(ctx, x + 10, y, h, "T " + fmt(f.t, 1) + " S", C.text, false); return; }
    x = annunciator(ctx, x + 10, y, h, "TP-" + s.n + "  " + s.info.test_point.title.toUpperCase(), C.text, false);
    x = annunciator(ctx, x + 10, y, h, "T " + fmt(f.t, 1) + " S", C.text, false);
    var state = {ready: "READY", paused: "PAUSED", reading: "RECORDER", debrief: "DEBRIEF",
                 error: "STOPPED", loading: "PREPARING"}[s.state];
    if (state) { annunciator(ctx, x + 10, y, h, state, C.amber, false); }
    var fld = f.field;
    if (fld && fld.distance >= 0 && fld.closing > 1) {
      var secs = fld.distance / fld.closing;
      var label = (fld.bearing === null ? fld.label.replace(/^.* core /, "CORE ") : "COLUMN") +
        " " + km(fld.distance).toUpperCase() + "  " + fmt(secs, 0) + " S";
      ctx.font = "700 14px " + FONT;
      var w = ctx.measureText(label).width + 24;
      annunciator(ctx, r.x + r.w - w, y, h, label, secs < 10 ? C.amber : C.text, false);
    }
  }

  function annunciator(ctx, x, y, h, label, color, boxed) {
    ctx.font = "700 14px " + FONT;
    var w = ctx.measureText(label).width + 24;
    ctx.fillStyle = C.panel;
    roundRect(ctx, x, y, w, h, 4);
    ctx.fill();
    ctx.lineWidth = boxed ? 2 : 1;
    ctx.strokeStyle = boxed ? color : C.edge;
    ctx.stroke();
    text(ctx, label, x + w / 2, y + h / 2 + 1, 14, color, "center", 700);
    return x + w;
  }

  function drawAttitude(ctx, r, f, lim) {
    var s = r.s, R = s / 2, cx = r.x + R, cy = r.y + R;
    var ppd = R / 20;  // pixels per degree of pitch: +/-20 deg from the centre to the edge
    var phi = f.bank * Math.PI / 180;
    ctx.save();
    roundRect(ctx, r.x, r.y, s, s, 12);
    ctx.clip();
    ctx.translate(cx, cy);
    ctx.rotate(-phi);
    var hy = f.pitch * ppd;  // nose up puts the horizon below the centre
    ctx.fillStyle = C.sky; ctx.fillRect(-2 * s, hy - 3 * s, 4 * s, 3 * s);
    ctx.fillStyle = C.ground; ctx.fillRect(-2 * s, hy, 4 * s, 3 * s);
    ctx.strokeStyle = C.horizon; ctx.lineWidth = 2;
    line(ctx, -2 * s, hy, 2 * s, hy);
    // The pitch ladder, clipped to a band about the centre as on a PFD.
    ctx.save();
    ctx.beginPath(); ctx.rect(-R * 0.62, -R * 0.72, R * 1.24, R * 1.44); ctx.clip();
    ctx.lineWidth = 1.6;
    for (var d = -40; d <= 40; d += 2.5) {
      if (d === 0) { continue; }
      var y = hy - d * ppd;
      if (Math.abs(y) > R * 0.8) { continue; }
      var half = d % 10 === 0 ? R * 0.26 : d % 5 === 0 ? R * 0.14 : R * 0.06;
      line(ctx, -half, y, half, y);
      if (d % 10 === 0) {
        text(ctx, String(Math.abs(d)), -half - 16, y + 1, 13, C.horizon, "center", 700);
        text(ctx, String(Math.abs(d)), half + 16, y + 1, 13, C.horizon, "center", 700);
      }
    }
    ctx.restore();
    ctx.restore();

    // The bank scale, fixed, and the roll pointer, which turns with the sky.
    ctx.save();
    ctx.translate(cx, cy);
    var ar = R * 0.84;
    ctx.strokeStyle = C.horizon; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(0, 0, ar, (-90 - 60) * Math.PI / 180, (-90 + 60) * Math.PI / 180); ctx.stroke();
    [-60, -45, -30, -20, -10, 10, 20, 30, 45, 60].forEach(function (b) {
      var a = (b - 90) * Math.PI / 180, len = Math.abs(b) % 30 === 0 ? 14 : 8;
      line(ctx, Math.cos(a) * ar, Math.sin(a) * ar, Math.cos(a) * (ar + len), Math.sin(a) * (ar + len));
    });
    ctx.fillStyle = C.horizon;
    tri(ctx, 0, -ar - 2, 9, -1);  // the zero index
    ctx.rotate(-phi);
    ctx.fillStyle = C.symbol;
    tri(ctx, 0, -ar + 2, 9, 1);   // the roll pointer
    ctx.restore();

    // The slip ball (n_y, not beta), at a fixed place under the symbol, as the
    // matplotlib panel draws it: a force to the right puts the ball left.
    var off = clamp(-f.ny / lim.slip_span, -1, 1) * R * 0.16;
    var by = cy + R * 0.86;
    ctx.strokeStyle = C.horizon; ctx.lineWidth = 1.5;
    line(ctx, cx - R * 0.16 - 8, by - 9, cx - R * 0.16 - 8, by + 9);
    line(ctx, cx + R * 0.16 + 8, by - 9, cx + R * 0.16 + 8, by + 9);
    ctx.fillStyle = C.symbol;
    ctx.beginPath(); ctx.arc(cx + off, by, 7, 0, 2 * Math.PI); ctx.fill();

    // The aircraft symbol: wings and a centre square, outlined for contrast.
    ctx.lineJoin = "round";
    [[C.glare, 7], [C.symbol, 4]].forEach(function (p) {
      ctx.strokeStyle = p[0]; ctx.lineWidth = p[1];
      line(ctx, cx - R * 0.56, cy, cx - R * 0.2, cy);
      line(ctx, cx - R * 0.2, cy, cx - R * 0.2, cy + R * 0.08);
      line(ctx, cx + R * 0.56, cy, cx + R * 0.2, cy);
      line(ctx, cx + R * 0.2, cy, cx + R * 0.2, cy + R * 0.08);
    });
    ctx.fillStyle = C.glare; ctx.fillRect(cx - 7, cy - 7, 14, 14);
    ctx.fillStyle = C.symbol; ctx.fillRect(cx - 5, cy - 5, 10, 10);
    frameBox(ctx, r.x, r.y, s, s, 12);
  }

  // A vertical tape: the scale slides past a fixed pointer, the value in a box.
  function drawTape(ctx, r, value, target, ppu, minor, major, side, unit, dp) {
    ctx.save();
    ctx.fillStyle = C.tape;
    roundRect(ctx, r.x, r.y, r.w, r.h, 6); ctx.fill();
    ctx.clip();
    var cy = r.y + r.h / 2;
    var span = r.h / 2 / ppu;
    var first = Math.floor((value - span) / minor) * minor;
    ctx.strokeStyle = C.text; ctx.lineWidth = 1.5;
    var edge = side === "left" ? r.x + r.w : r.x;
    var dir = side === "left" ? -1 : 1;
    for (var v = first; v <= value + span; v += minor) {
      var y = cy - (v - value) * ppu;
      var isMajor = Math.abs(v / major - Math.round(v / major)) < 1e-6;
      line(ctx, edge, y, edge + dir * (isMajor ? 14 : 8), y);
      if (isMajor) {
        text(ctx, group(v), edge + dir * 20, y + 1, 14, C.text, side === "left" ? "right" : "left", 700);
      }
    }
    // The start condition, a magenta bug.
    var ty = cy - (target - value) * ppu;
    if (ty > r.y - 10 && ty < r.y + r.h + 10) {
      ctx.fillStyle = C.magenta;
      ctx.fillRect(side === "left" ? r.x + r.w - 5 : r.x, ty - 9, 5, 18);
    }
    ctx.restore();
    frameBox(ctx, r.x, r.y, r.w, r.h, 6);
    // The value box.
    var bh = 38, bw = r.w + 4, bx = r.x - 2;
    ctx.fillStyle = C.glare;
    roundRect(ctx, bx, cy - bh / 2, bw, bh, 4); ctx.fill();
    ctx.strokeStyle = C.text; ctx.lineWidth = 2; ctx.stroke();
    text(ctx, group(value, dp), bx + bw / 2, cy + 1, 21, C.text, "center", 700);
    text(ctx, unit, r.x + r.w / 2, r.y - 10, 12, C.dim, "center", 700);
    text(ctx, group(target), r.x + r.w / 2, r.y + r.h + 16, 12.5, C.magenta, "center", 700);
  }

  function drawVsi(ctx, r, vs, span) {
    ctx.fillStyle = C.tape;
    roundRect(ctx, r.x, r.y, r.w, r.h, 6); ctx.fill();
    frameBox(ctx, r.x, r.y, r.w, r.h, 6);
    var cy = r.y + r.h / 2, k = (r.h / 2 - 10) / span;
    ctx.strokeStyle = C.dim; ctx.lineWidth = 1.2;
    [-1, -0.5, 0, 0.5, 1].forEach(function (u) {
      var y = cy - u * span * k;
      line(ctx, r.x + 4, y, r.x + (u === 0 ? 18 : 12), y);
    });
    var y = cy - clamp(vs, -span, span) * k;
    ctx.strokeStyle = C.text; ctx.lineWidth = 3;
    line(ctx, r.x + 6, cy, r.x + r.w - 4, y);
    text(ctx, "V/S", r.x + r.w / 2, r.y - 10, 12, C.dim, "center", 700);
    text(ctx, signed(vs, 1), r.x + r.w / 2, r.y + r.h + 16, 13, C.text, "center", 700);
  }

  function drawHeading(ctx, r, hdg, target) {
    ctx.save();
    ctx.fillStyle = C.tape;
    roundRect(ctx, r.x, r.y, r.w, r.h, 6); ctx.fill();
    ctx.clip();
    var cx = r.x + r.w / 2, ppd = r.w / 60, ly = r.y + 29;
    ctx.strokeStyle = C.text; ctx.lineWidth = 1.5;
    var first = Math.floor((hdg - 32) / 5) * 5;
    for (var d = first; d <= hdg + 32; d += 5) {
      var x = cx + dAngle(d, hdg) * ppd;
      var major = ((d % 10) + 10) % 10 === 0;
      line(ctx, x, r.y, x, r.y + (major ? 12 : 7));
      if (major && Math.abs(x - cx) > 40) {
        var lab = String(((d % 360) + 360) % 360).padStart(3, "0");
        text(ctx, lab, x, ly, 13, C.text, "center", 700);
      }
    }
    var tx = cx + dAngle(target, hdg) * ppd;
    ctx.fillStyle = C.magenta; ctx.fillRect(tx - 8, r.y, 16, 5);
    ctx.restore();
    frameBox(ctx, r.x, r.y, r.w, r.h, 6);
    ctx.fillStyle = C.glare;
    roundRect(ctx, cx - 30, ly - 12, 60, 24, 3); ctx.fill();
    ctx.strokeStyle = C.text; ctx.lineWidth = 1.5; ctx.stroke();
    text(ctx, String(Math.round(hdg) % 360).padStart(3, "0"), cx, ly + 1, 15, C.text, "center", 700);
    ctx.fillStyle = C.symbol; tri(ctx, cx, r.y + 2, 7, 1);
  }

  // The flight-test instruments: what the test point is about.
  function drawInstruments(ctx, r, s, f, lim) {
    var x = r.x, w = r.w, y = r.y;
    // Load factor: a vertical scale with the peaks held.
    var gh = Math.max(150, r.h * 0.46);
    label(ctx, "LOAD FACTOR", x, y);
    text(ctx, signed(f.nz, 2) + " g", x + w, y - 3, 22, C.text, "right", 700, FONT, "top");
    var sy = y + 34, sh = gh - 34, sx = x + 26, sw = 26;
    var lo = lim.nz_range[0], hi = lim.nz_range[1];
    var yOf = function (g) { return sy + (hi - clamp(g, lo, hi)) / (hi - lo) * sh; };
    ctx.fillStyle = C.tape; roundRect(ctx, sx, sy, sw, sh, 4); ctx.fill();
    frameBox(ctx, sx, sy, sw, sh, 4);
    ctx.strokeStyle = C.dim; ctx.lineWidth = 1;
    for (var g = Math.ceil(lo); g <= hi; g += 1) {
      line(ctx, sx - 6, yOf(g), sx, yOf(g));
      text(ctx, (g > 0 ? "+" : g < 0 ? "−" : "") + Math.abs(g), sx - 10, yOf(g) + 1, 12, C.dim, "right", 700);
    }
    ctx.strokeStyle = C.text; ctx.lineWidth = 1.5;
    line(ctx, sx, yOf(1), sx + sw, yOf(1));
    // The bar from 1 g (level flight) to now.
    ctx.fillStyle = C.trace;
    var y1 = yOf(1), yn = yOf(f.nz);
    ctx.fillRect(sx + 6, Math.min(y1, yn), sw - 12, Math.max(2, Math.abs(yn - y1)));
    // The peaks since the start, held, labelled once they leave level flight.
    if (s.peak) {
      ctx.fillStyle = C.symbol;
      [[s.peak.max, "peak "], [s.peak.min, "low "]].forEach(function (p) {
        if (Math.abs(p[0] - 1) < 0.05) { return; }
        tri(ctx, sx + sw + 9, yOf(p[0]), 6, 0, true);
        text(ctx, p[1] + signed(p[0], 2), sx + sw + 20, yOf(p[0]) + 1, 12.5, C.symbol, "left", 700);
      });
    }
    text(ctx, "1 g: level flight", x, sy + sh + 8, 11.5, C.dim, "left", 400, FONT, "top");

    // Angle of attack against the model's linear band.
    y = sy + sh + 34;
    label(ctx, "ANGLE OF ATTACK", x, y);
    var aa = Math.abs(f.alpha);
    var band = aa >= lim.alpha_invalid_deg ? "invalid" : aa >= lim.alpha_linear_deg ? "marginal" : "linear";
    var bandColor = band === "invalid" ? C.red : band === "marginal" ? C.amber : C.text;
    text(ctx, signed(f.alpha, 1) + "° " + band, x + w, y, 14, bandColor, "right", 700, FONT, "top");
    var ay = y + 26, ahh = 14, span = 15;
    var xOf = function (a) { return x + clamp(a, 0, span) / span * w; };
    ctx.fillStyle = C.tape; ctx.fillRect(x, ay, w, ahh);
    ctx.fillStyle = "rgba(255,176,32,0.35)";
    ctx.fillRect(xOf(lim.alpha_linear_deg), ay, xOf(lim.alpha_invalid_deg) - xOf(lim.alpha_linear_deg), ahh);
    ctx.fillStyle = "rgba(255,90,78,0.4)";
    ctx.fillRect(xOf(lim.alpha_invalid_deg), ay, xOf(span) - xOf(lim.alpha_invalid_deg), ahh);
    ctx.strokeStyle = C.edge; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, ay + 0.5, w - 1, ahh - 1);
    ctx.fillStyle = C.text; ctx.fillRect(xOf(aa) - 1.5, ay - 4, 3, ahh + 8);
    text(ctx, "0", x, ay + ahh + 12, 11.5, C.dim, "left", 700);
    text(ctx, fmt(lim.alpha_linear_deg, 0) + "°", xOf(lim.alpha_linear_deg), ay + ahh + 12, 11.5, C.dim, "center", 700);
    text(ctx, fmt(span, 0) + "°", x + w, ay + ahh + 12, 11.5, C.dim, "right", 700);

    // Wind, north up: the arrow points where the air is going.
    y = ay + ahh + 34;
    label(ctx, "WIND", x, y);
    var wn = f.wind.north, we = f.wind.east, sp = Math.hypot(wn, we);
    text(ctx, fmt(sp, 1) + " m/s, " + (f.wind.up >= 0 ? "up " : "down ") + fmt(Math.abs(f.wind.up), 1),
         x + w, y, 13, C.text, "right", 700, FONT, "top");
    var wr = Math.min(34, (r.y + r.h - y - 40) / 2);
    if (wr > 12) {
      var wx = x + wr + 4, wy = y + 26 + wr;
      ctx.strokeStyle = C.edge; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.arc(wx, wy, wr, 0, 2 * Math.PI); ctx.stroke();
      text(ctx, "N", wx, wy - wr - 8, 11, C.dim, "center", 700);
      if (sp > 1e-6) {
        var k = Math.min(sp / lim.wind_span, 1) * wr;
        ctx.strokeStyle = C.trace; ctx.lineWidth = 2.5;
        line(ctx, wx, wy, wx + we / sp * k, wy - wn / sp * k);
        ctx.fillStyle = C.trace; ctx.beginPath(); ctx.arc(wx + we / sp * k, wy - wn / sp * k, 3.5, 0, 2 * Math.PI); ctx.fill();
      }
      // Gust rate: p, q, r bars. No instrument senses these; the model does.
      var gx = wx + wr + 18, gw = x + w - gx;
      if (gw > 60) {
        text(ctx, "GUST RATE", gx, y + 26, 11.5, C.dim, "left", 700, FONT, "top");
        text(ctx, "SIM TRUTH", gx, y + 40, 11.5, C.red, "left", 700, FONT, "top");
        ["p", "q", "r"].forEach(function (n, i) {
          var yy = y + 62 + i * 15, mid = gx + 12 + (gw - 12) / 2;
          var v = clamp(f.gust[i] / lim.gust_span, -1, 1) * (gw - 12) / 2;
          text(ctx, n, gx, yy + 1, 12, C.dim, "left", 700);
          ctx.fillStyle = C.tape; ctx.fillRect(gx + 12, yy - 4, gw - 12, 8);
          ctx.fillStyle = C.trace; ctx.fillRect(Math.min(mid, mid + v), yy - 4, Math.max(1.5, Math.abs(v)), 8);
          ctx.fillStyle = C.dim; ctx.fillRect(mid - 0.5, yy - 6, 1, 12);
        });
      }
    }
  }

  // The last STRIP_S seconds of load factor, sliding past a fixed frame.
  function drawStrip(ctx, r, s, f, lim) {
    ctx.fillStyle = C.panel;
    roundRect(ctx, r.x, r.y, r.w, r.h, 6); ctx.fill();
    frameBox(ctx, r.x, r.y, r.w, r.h, 6);
    var px = r.x + 150, pw = r.w - 150 - 14, py = r.y + 12, ph = r.h - 34;
    label(ctx, "LOAD FACTOR", r.x + 14, r.y + 12);
    text(ctx, "last " + STRIP_S + " s", r.x + 14, r.y + 30, 12, C.dim, "left", 400, FONT, "top");
    var lo = Math.min(0, Math.floor(s.peak ? s.peak.min : 0)), hi = Math.max(2, Math.ceil(s.peak ? s.peak.max : 2));
    var yOf = function (g) { return py + (hi - g) / (hi - lo) * ph; };
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
    for (var g = lo; g <= hi; g += 1) {
      line(ctx, px, yOf(g), px + pw, yOf(g));
      text(ctx, (g > 0 ? "+" : g < 0 ? "−" : "") + Math.abs(g) + " g", px - 8, yOf(g) + 1, 11.5, C.dim, "right", 700);
    }
    for (var sec = 0; sec <= STRIP_S; sec += 10) {
      var xx = px + pw - sec / STRIP_S * pw;
      line(ctx, xx, py, xx, py + ph);
      text(ctx, sec === 0 ? "now" : "−" + sec + " s", xx, py + ph + 12, 11, C.dim,
           sec === STRIP_S ? "left" : sec === 0 ? "right" : "center", 700);
    }
    ctx.strokeStyle = C.dim; ctx.lineWidth = 1.2;
    line(ctx, px, yOf(1), px + pw, yOf(1));
    if (s.trace.length > 1) {
      ctx.save();
      ctx.beginPath(); ctx.rect(px, py - 2, pw, ph + 4); ctx.clip();
      ctx.strokeStyle = C.trace; ctx.lineWidth = 2; ctx.lineJoin = "round";
      ctx.beginPath();
      s.trace.forEach(function (p, i) {
        var xx2 = px + pw - (f.t - p.t) / STRIP_S * pw, yy = yOf(p.nz);
        if (i === 0) { ctx.moveTo(xx2, yy); } else { ctx.lineTo(xx2, yy); }
      });
      ctx.stroke();
      ctx.restore();
    }
  }

  // ---- the debrief ------------------------------------------------------------------
  function debrief(s, sum) {
    var tp = s.info.test_point;
    var a = sum.alpha_max, rec = sum.recovery, sev = sum.severity;
    var aNote = a.band === "linear" ? "the linear aerodynamics hold below " + fmt(s.info.limits.alpha_linear_deg, 0) + "°"
      : a.band === "marginal" ? "past " + fmt(s.info.limits.alpha_linear_deg, 0) + "° the linear aerodynamics stop holding"
      : "past " + fmt(s.info.limits.alpha_invalid_deg, 0) + "° the model's lift is not real: this flight proves nothing there";
    var recText = rec.passed === true ? "Flown inside the conditions the 747's derivatives were recovered at."
      : rec.passed === false ? "Flown outside the conditions the 747's derivatives were recovered at: " + rec.detail
      : "Report: " + rec.detail;
    var sevText = sum.severity_band === "too short to rate" ? "Too short to rate"
      : cap(sum.severity_band);
    var sevNote = isFinite(sev.value) ? "RMS normal load " + fmt(sev.value, 2) + " g over 5 s. Moderate from 0.2 g, " +
      "severe from 0.3 g (Misaka 2008)." : "The index needs at least 5 s of flight.";
    var ac = sum.altitude_change;
    var html =
      '<div class="fly-debrief" role="dialog" aria-label="Results of test point ' + escapeHtml(s.n) + '">' +
      '<div class="fly-debrief-head"><h2>TP-' + escapeHtml(s.n) + " · " + escapeHtml(tp.title) + " · results</h2>" +
      "<span>" + fmt(sum.duration, 1) + " s flown · Boeing 747 model</span></div>" +
      '<div class="fly-debrief-top"><div class="fly-peak-row">' +
      '<span class="tc-field">Peak load factor</span><div>' +
      '<div class="fly-peak-big">' + signed(sum.nz_max.value, 2) + " g</div>" +
      '<p class="fly-peak-what">at ' + fmt(sum.nz_max.t, 1) + " s. Lowest " + signed(sum.nz_min.value, 2) +
      " g at " + fmt(sum.nz_min.t, 1) + " s. Level flight is +1 g.</p>" +
      '<p class="fly-peak-what">Read from every physics step. The panel samples 25 times a second, ' +
      "so its peak can be a little lower.</p></div></div>" +
      '<dl class="fly-facts-grid">' +
      '<dt class="tc-field">Turbulence</dt><dd><b>' + escapeHtml(sevText) + "</b><small>" + escapeHtml(sevNote) + "</small></dd>" +
      '<dt class="tc-field">Bank</dt><dd>up to ' + fmt(sum.bank_max, 1) + "°</dd>" +
      '<dt class="tc-field">Height</dt><dd>' + signed(ac.min, 0) + " m to " + signed(ac.max, 0) + " m; ended " + signed(ac.end, 0) + " m</dd>" +
      '<dt class="tc-field">Angle of attack</dt><dd' + (a.band === "linear" ? "" : ' class="is-limit"') + ">up to " +
      fmt(Math.abs(a.value), 1) + "°, " + a.band + "<small>" + escapeHtml(aNote) + "</small></dd>" +
      '<dt class="tc-field">Validity</dt><dd' + (rec.passed === false ? ' class="is-limit"' : "") + ">" + escapeHtml(recText) + "</dd>" +
      "</dl></div>" +
      '<div class="fly-recorder"><div class="fly-recorder-title"><span class="tc-field">Recorder</span>' +
      '<span class="fly-peak-what">every physics step, 50 a second' + (sum.events.length ? " · blue lines: where you met the field" : "") +
      (sum.series.autopilot.indexOf(true) >= 0 ? " · shaded: autopilot on" : "") +
      "</span></div><canvas></canvas></div>" +
      '<div class="fly-note-row"><button type="button" class="fly-btn fly-btn-primary" data-act="again">Fly TP-' + escapeHtml(s.n) + " again</button>" +
      '<button type="button" class="fly-btn" data-act="card">Back to the test card</button>' +
      '<button type="button" class="fly-btn" data-act="workbench">Engineering mode</button></div></div>';
    s.overlay.classList.add("is-dim");
    s.overlay.innerHTML = html;
    var canvas = s.overlay.querySelector(".fly-recorder canvas");
    var drawIt = function () { recorder(canvas, sum); };
    requestAnimationFrame(drawIt);
    focusIn(s, '[data-act="again"]');
  }

  // The recorder: load factor, height change and bank on one time axis.
  function recorder(canvas, sum) {
    var box = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(box.width * dpr); canvas.height = Math.round(box.height * dpr);
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var W = box.width, H = box.height, sr = sum.series;
    ctx.fillStyle = C.paper; ctx.fillRect(0, 0, W, H);
    var left = 172, right = 18, top = 14, bottom = 26, gap = 10;
    var pw = W - left - right, ph = (H - top - bottom - 2 * gap) / 3;
    var t0 = 0, t1 = Math.max(1, sum.duration);
    var xOf = function (t) { return left + (t - t0) / (t1 - t0) * pw; };
    // Autopilot spans, under everything.
    ctx.fillStyle = C.apWash;
    var startAp = null;
    sr.t.forEach(function (t, i) {
      if (sr.autopilot[i] && startAp === null) { startAp = t; }
      if ((!sr.autopilot[i] || i === sr.t.length - 1) && startAp !== null) {
        ctx.fillRect(xOf(startAp), top, xOf(t) - xOf(startAp), H - top - bottom);
        startAp = null;
      }
    });
    var panels = [
      {key: "nz", name: "Load factor", unit: "g", ref: 1, fmtv: function (v) { return signed(v, 1); }},
      {key: "altitude", name: "Height change", unit: "m", ref: 0, fmtv: function (v) { return signed(v, 0); }},
      {key: "bank", name: "Bank", unit: "°", ref: 0, fmtv: function (v) { return signed(v, 0); }}
    ];
    panels.forEach(function (p, k) {
      var y0 = top + k * (ph + gap), vals = sr[p.key];
      var lo = Math.min.apply(null, vals.concat([p.ref])), hi = Math.max.apply(null, vals.concat([p.ref]));
      var pad = Math.max((hi - lo) * 0.12, p.key === "nz" ? 0.1 : 2);
      lo -= pad; hi += pad;
      var yOf = function (v) { return y0 + (hi - v) / (hi - lo) * ph; };
      ctx.strokeStyle = C.rule; ctx.lineWidth = 1;
      ctx.strokeRect(left + 0.5, y0 + 0.5, pw - 1, ph - 1);
      ctx.strokeStyle = "#9aa1a9"; line(ctx, left, yOf(p.ref), left + pw, yOf(p.ref));
      text(ctx, p.name.toUpperCase(), 10, y0 + 12, 11.5, C.ink2, "left", 700);
      text(ctx, p.unit, 10, y0 + 28, 12, C.ink2, "left", 400);
      text(ctx, p.fmtv(hi), left - 8, y0 + 8, 11.5, C.ink2, "right", 700);
      text(ctx, p.fmtv(lo), left - 8, y0 + ph - 8, 11.5, C.ink2, "right", 700);
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5; ctx.lineJoin = "round";
      ctx.beginPath();
      sr.t.forEach(function (t, i) {
        if (i === 0) { ctx.moveTo(xOf(t), yOf(vals[i])); } else { ctx.lineTo(xOf(t), yOf(vals[i])); }
      });
      ctx.stroke();
      if (p.key === "nz") {
        var px = xOf(sum.nz_max.t), py = yOf(sum.nz_max.value);
        ctx.strokeStyle = C.pencil; ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.arc(px, py, 9, 0, 2 * Math.PI); ctx.stroke();
        ctx.font = "20px " + HAND; ctx.fillStyle = C.pencil;
        ctx.textAlign = px > left + pw - 110 ? "right" : "left"; ctx.textBaseline = "middle";
        ctx.fillText(signed(sum.nz_max.value, 2) + " g", px + (ctx.textAlign === "right" ? -16 : 16), Math.max(y0 + 12, py - 4));
      }
    });
    // Where the flight met the field.
    ctx.strokeStyle = C.stamp; ctx.lineWidth = 1.5;
    sum.events.forEach(function (e) {
      var x = xOf(e.t);
      line(ctx, x, top, x, H - bottom);
      var nearEnd = x > left + pw - 90;
      text(ctx, e.label.replace(/^passed .* core /, "core ").replace("closest to the column centre", "column centre"),
           x + (nearEnd ? -4 : 4), H - bottom - 10, 11.5, C.stamp, nearEnd ? "right" : "left", 700);
    });
    for (var t = 0; t <= t1; t += niceStep(t1)) {
      text(ctx, fmt(t, 0) + " s", xOf(t), H - 10, 11.5, C.ink2, "center", 700);
    }
  }

  // ---- small drawing helpers -----------------------------------------------------------
  function text(ctx, str, x, y, px, color, align, weight, face, baseline) {
    ctx.font = (weight || 400) + " " + px + "px " + (face || FONT);
    ctx.fillStyle = color;
    ctx.textAlign = align || "left";
    ctx.textBaseline = baseline || "middle";
    ctx.fillText(str, x, y);
  }
  function label(ctx, str, x, y) { text(ctx, str, x, y, 12, C.dim, "left", 700, FONT, "top"); }
  function line(ctx, x0, y0, x1, y1) { ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); }
  function tri(ctx, x, y, size, dir, sideways) {
    ctx.beginPath();
    if (sideways) {
      ctx.moveTo(x - size, y); ctx.lineTo(x + size * 0.4, y - size * 0.8); ctx.lineTo(x + size * 0.4, y + size * 0.8);
    } else {
      ctx.moveTo(x, y); ctx.lineTo(x - size, y + dir * size * 1.4); ctx.lineTo(x + size, y + dir * size * 1.4);
    }
    ctx.closePath(); ctx.fill();
  }
  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }
  function frameBox(ctx, x, y, w, h, r) {
    ctx.strokeStyle = C.edge; ctx.lineWidth = 1.5;
    roundRect(ctx, x, y, w, h, r); ctx.stroke();
  }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function dAngle(a, b) { return ((a - b + 540) % 360) - 180; }
  function fmt(v, dp) { return Number(v).toFixed(dp); }
  function signed(v, dp) { var s = Math.abs(v).toFixed(dp); return (v < 0 && Number(s) !== 0 ? "−" : "+") + s; }
  function group(v, dp) { return Number(v).toLocaleString("en-GB", {maximumFractionDigits: dp || 0, minimumFractionDigits: dp || 0}); }
  function km(m) { return m >= 1000 ? (m / 1000).toFixed(1) + " km" : Math.round(m) + " m"; }
  function cap(str) { return str.charAt(0).toUpperCase() + str.slice(1); }
  function niceStep(span) { return span > 240 ? 60 : span > 120 ? 30 : span > 50 ? 10 : 5; }
  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, function (c) {
      return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c];
    });
  }
})();
