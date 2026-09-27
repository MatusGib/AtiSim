/* Resizable splitters for the Model Builder panels.

   A splitter is an element with class "ati-split", role "separator" and
   data-var naming the CSS custom property it sets on its grid container
   (data-min / data-max in px). Pointer drag and arrow keys both work, so the
   splitters are keyboard reachable. Delegated from the document because the
   router swaps page bodies in and out. */
(function () {
  "use strict";
  var active = null;

  function limits(el) {
    return [parseFloat(el.dataset.min || "120"), parseFloat(el.dataset.max || "900")];
  }

  function current(el) {
    var host = el.closest("[data-split-host]");
    var value = getComputedStyle(host).getPropertyValue(el.dataset.var);
    return [host, parseFloat(value) || 0];
  }

  function set(el, host, value) {
    var lim = limits(el);
    var v = Math.max(lim[0], Math.min(lim[1], value));
    host.style.setProperty(el.dataset.var, v + "px");
    el.setAttribute("aria-valuenow", String(Math.round(v)));
    window.dispatchEvent(new Event("resize"));
  }

  document.addEventListener("pointerdown", function (e) {
    var el = e.target.closest && e.target.closest(".ati-split[data-var]");
    if (!el) return;
    var hc = current(el);
    active = { el: el, host: hc[0], start: hc[1], x: e.clientX, y: e.clientY,
               sign: parseFloat(el.dataset.sign || "1"),
               vertical: el.getAttribute("aria-orientation") === "vertical" };
    el.classList.add("is-dragging");
    el.setPointerCapture(e.pointerId);
    e.preventDefault();
  });

  document.addEventListener("pointermove", function (e) {
    if (!active) return;
    var delta = active.vertical ? e.clientX - active.x : e.clientY - active.y;
    set(active.el, active.host, active.start + active.sign * delta);
  });

  function end() {
    if (!active) return;
    active.el.classList.remove("is-dragging");
    active = null;
  }
  document.addEventListener("pointerup", end);
  document.addEventListener("pointercancel", end);

  document.addEventListener("keydown", function (e) {
    var el = e.target.closest && e.target.closest(".ati-split[data-var]");
    if (!el) return;
    var vertical = el.getAttribute("aria-orientation") === "vertical";
    var grow = vertical ? "ArrowRight" : "ArrowUp";
    var shrink = vertical ? "ArrowLeft" : "ArrowDown";
    if (e.key !== grow && e.key !== shrink) return;
    var hc = current(el);
    set(el, hc[0], hc[1] + (e.key === grow ? 16 : -16));
    e.preventDefault();
  });
})();
