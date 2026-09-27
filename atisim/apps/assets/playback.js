/* Diagnostics: the cursor and playback, in the browser.

   moveCursor moves the named cursor lines ("cursor", devfigures._cursor) of
   the Diagnostics figures to time t, and draws nothing else again. Plotly
   computes a figure again to move a line, which takes tens of milliseconds, so
   while playing it moves only the figures in view.

   tick is one step of playback. It moves the cursor by the wall clock, not by
   ticks: a tick that comes late moves it further, so 1x is flight time at the
   speed of real time. It moves the lines itself and sends the time to the
   server (diag-cursor) at most once per config.commit seconds, so the panels
   that the server draws (the scene, the dominant term) follow without a queue. */
window.dash_clientside = Object.assign({}, window.dash_clientside, {
  atisim: {
    moveCursor: function (t, config, paused) {
      // Playing, only the figures in view move, and a tick whose figures are
      // still drawing the last move is skipped: the browser keeps up. Paused,
      // every figure moves.
      var ns = window.dash_clientside;
      if (t === null || t === undefined || !window.Plotly || !config) {
        return ns.no_update;
      }
      var playing = paused === false;
      if (playing && ns.atisim.busy) { return t; }
      var drawn = [];
      config.graphs.forEach(function (id) {
        var host = document.getElementById(id);
        var gd = host && (host.classList.contains("js-plotly-plot") ? host
                          : host.querySelector(".js-plotly-plot"));
        if (!gd || !gd.layout || !gd.layout.shapes) { return; }
        if (playing) {
          var box = gd.getBoundingClientRect();
          if (box.bottom < 0 || box.top > window.innerHeight) { return; }
        }
        var update = {};
        gd.layout.shapes.forEach(function (shape, i) {
          if (shape.name === "cursor" && shape.x0 !== t) {
            update["shapes[" + i + "].x0"] = t;
            update["shapes[" + i + "].x1"] = t;
          }
        });
        if (Object.keys(update).length) { drawn.push(window.Plotly.relayout(gd, update)); }
      });
      if (drawn.length) {
        ns.atisim.busy = true;
        Promise.all(drawn).then(function () { ns.atisim.busy = false; },
                                function () { ns.atisim.busy = false; });
      }
      return t;
    },

    tick: function (_n, played, cursor, speed, clock, bounds, config) {
      var ns = window.dash_clientside;
      var now = Date.now() / 1000;
      var last = clock || {};
      var from = last.tick ? played : cursor;
      if (from === null || from === undefined) { from = bounds[0]; }
      var elapsed = last.tick ? Math.min(Math.max(now - last.tick, 0), 2) : config.tick;
      var t = from + (config.speeds[speed] || 1) * elapsed;
      if (t > bounds[1]) { t = bounds[0]; }
      ns.atisim.moveCursor(t, config, false);
      var commit = !last.commit || now - last.commit >= config.commit;
      return [t, commit ? t : ns.no_update, {tick: now, commit: commit ? now : last.commit}];
    }
  }
});
