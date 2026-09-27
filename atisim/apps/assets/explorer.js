/* The explorer (atisim/apps/explorer.py), in the browser.

   The router rebuilds the tree from the URL on every page change; this keeps
   what the reader did to it:

     - which groups are open, for the session (the group holding the page on
       screen is always open);
     - the filter: typing shows every matching item in every group, and hides
       the groups with none; Escape clears it; "/" focuses it from anywhere;
     - the explorer hidden or shown (the button at the header's left), kept
       across visits, on <html> so no re-render of the app can undo it;
     - aria-current on the item for the page on screen (dcc.Link cannot carry
       it). */
(function () {
  "use strict";
  var OPEN = "atisim.explorer.open";
  var HIDDEN = "atisim.explorer.hidden";
  var quietUntil = 0;  // toggles we cause ourselves are not the reader's

  function read(k) {
    try { return JSON.parse(sessionStorage.getItem(k) || "{}"); } catch (e) { return {}; }
  }
  function write(k, v) {
    try { sessionStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* private window */ }
  }
  function quiet() { quietUntil = performance.now() + 250; }

  // ---- hidden or shown ---------------------------------------------------------
  function setHidden(hidden) {
    document.documentElement.dataset.explorer = hidden ? "hidden" : "shown";
    var btn = document.getElementById("explorer-toggle");
    if (btn) { btn.setAttribute("aria-expanded", hidden ? "false" : "true"); }
    try { localStorage.setItem(HIDDEN, hidden ? "1" : "0"); } catch (e) { /* ignore */ }
    window.dispatchEvent(new Event("resize"));  // Plotly refits to the new width
  }
  try {
    if (localStorage.getItem(HIDDEN) === "1") { document.documentElement.dataset.explorer = "hidden"; }
  } catch (e) { /* ignore */ }

  document.addEventListener("click", function (e) {
    if (e.target.closest && e.target.closest("#explorer-toggle")) {
      setHidden(document.documentElement.dataset.explorer !== "hidden");
    }
  });

  // ---- open groups ----------------------------------------------------------------
  document.addEventListener("toggle", function (e) {
    var d = e.target;
    if (!d.matches || !d.matches("#explorer details[data-key]")) { return; }
    if (performance.now() < quietUntil || filtering()) { return; }
    var open = read(OPEN);
    open[d.dataset.key] = d.open;
    write(OPEN, open);
  }, true);

  function restoreOpen(tree) {
    var open = read(OPEN);
    quiet();
    tree.querySelectorAll("details[data-key]").forEach(function (d) {
      if (d.querySelector(".is-active")) { d.open = true; }
      else if (d.dataset.key in open) { d.open = open[d.dataset.key]; }
    });
  }

  // ---- the filter ------------------------------------------------------------------
  function input() { return document.getElementById("explorer-filter"); }
  function filtering() { var i = input(); return !!(i && i.value.trim()); }

  function filter() {
    var tree = document.getElementById("explorer");
    var box = input();
    if (!tree || !box) { return; }
    var q = box.value.trim().toLowerCase();
    tree.classList.toggle("is-filtering", !!q);
    var hits = 0;
    tree.querySelectorAll(".ati-xitem").forEach(function (a) {
      var text = (a.textContent + " " + (a.getAttribute("title") || "")).toLowerCase();
      var hit = !q || text.indexOf(q) >= 0;
      a.hidden = !hit;
      if (hit && q) { hits += 1; }
    });
    quiet();
    tree.querySelectorAll("details").forEach(function (d) {
      if (!q) { d.hidden = false; return; }
      var visible = d.querySelector(".ati-xitem:not([hidden])");
      d.hidden = !visible;
      if (visible) { d.open = true; }
    });
    var none = tree.querySelector(".ati-xnomatch");
    if (none) { none.hidden = !q || hits > 0; }
    if (!q) { restoreOpen(tree); }
  }

  document.addEventListener("input", function (e) {
    if (e.target && e.target.id === "explorer-filter") { filter(); }
  });
  document.addEventListener("keydown", function (e) {
    var box = input();
    if (!box) { return; }
    if (e.target === box && e.key === "Escape") {
      // Through the native setter and an input event, so React's copy agrees.
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(box, "");
      box.dispatchEvent(new Event("input", {bubbles: true}));
      return;
    }
    var tag = e.target && e.target.tagName;
    var typing = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" ||
      (e.target && e.target.isContentEditable);
    if (e.key === "/" && !typing && !e.ctrlKey && !e.metaKey && !e.altKey) {
      if (document.documentElement.dataset.explorer === "hidden") { setHidden(false); }
      e.preventDefault();
      box.focus();
      box.select();
    }
  });

  // ---- after every rebuild ------------------------------------------------------------
  var seen = null;
  function rebuilt() {
    var tree = document.getElementById("explorer");
    if (!tree || tree.firstElementChild === seen) { return; }
    seen = tree.firstElementChild;
    restoreOpen(tree);
    tree.querySelectorAll(".ati-xitem").forEach(function (a) {
      if (a.classList.contains("is-active")) { a.setAttribute("aria-current", "page"); }
      else { a.removeAttribute("aria-current"); }
    });
    var active = tree.querySelector(".ati-xitem.is-active");
    if (active) { active.scrollIntoView({block: "nearest"}); }
    if (filtering()) { filter(); }
    var btn = document.getElementById("explorer-toggle");
    if (btn) {
      btn.setAttribute("aria-expanded",
                       document.documentElement.dataset.explorer === "hidden" ? "false" : "true");
    }
  }
  new MutationObserver(rebuilt).observe(document.documentElement, {childList: true, subtree: true});
})();
