// Perception walker. Runs inside ONE frame document and returns the visible,
// meaningful elements as a flat list. It deliberately does not rely on ids,
// data-* attributes, ARIA, or test hooks: role and accessible name are
// computed from tag semantics, wrapping/for labels, and - the legacy-table
// case - the text of the cell to the left of a control.
//
// Output shape per element:
//   {role, name, tag, bbox:{x,y,w,h} (frame-relative CSS px), structural,
//    row_ctx:{row_index, headers, cells, table_path}|null, attrs:{...}}
() => {
  const MAX_NAME = 160;
  const root = document.body || document.documentElement;
  if (!root) return { elements: [], text: "", title: document.title || "", url: location.href };

  const INLINE = new Set(["b", "i", "u", "font", "span", "strong", "em", "small", "big",
    "sup", "sub", "tt", "code", "br", "wbr", "img", "label", "nobr"]);
  const SKIP = new Set(["script", "style", "noscript", "option", "optgroup", "hr", "tbody",
    "thead", "tfoot", "tr", "table", "form", "html", "head", "frameset", "frame", "iframe",
    "title", "meta", "link", "colgroup", "col", "input_hidden"]);
  const INTERACTIVE = new Set(["link", "button", "textbox", "checkbox", "radio", "combobox"]);

  const norm = (s) => (s || "").replace(/\s+/g, " ").trim();

  const isVisible = (el) => {
    const st = getComputedStyle(el);
    if (st.display === "none" || st.visibility === "hidden" || st.opacity === "0") return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  };

  const nthPath = (el) => {
    const parts = [];
    while (el && el.nodeType === 1 && el.tagName.toLowerCase() !== "html") {
      const tag = el.tagName.toLowerCase();
      let i = 1, s = el;
      while ((s = s.previousElementSibling)) if (s.tagName === el.tagName) i++;
      parts.unshift(tag + ":nth-of-type(" + i + ")");
      el = el.parentElement;
    }
    return "html>" + parts.join(">");
  };

  // True if some non-inline descendant, reached through inline wrappers only,
  // carries text of its own. Such an element is not the innermost block for
  // that text, so we do not emit it. Without the recursion a <body><font>
  // wrapper (ubiquitous in this era of markup) makes <body> look like the
  // innermost block and emits the whole page as one element.
  const hasBlockChildWithText = (el) => {
    for (const c of el.children) {
      const t = c.tagName.toLowerCase();
      if (INLINE.has(t)) {
        if (hasBlockChildWithText(c)) return true;
        continue;
      }
      if (norm(c.innerText)) return true;
    }
    return false;
  };

  const roleOf = (el) => {
    const explicit = el.getAttribute("role");
    if (explicit) return explicit;
    const tag = el.tagName.toLowerCase();
    if (tag === "a") return el.hasAttribute("href") ? "link" : "text";
    if (tag === "button") return "button";
    if (tag === "input") {
      const t = (el.getAttribute("type") || "text").toLowerCase();
      if (t === "hidden") return null;
      if (["submit", "button", "reset", "image"].includes(t)) return "button";
      if (t === "checkbox") return "checkbox";
      if (t === "radio") return "radio";
      return "textbox";
    }
    if (tag === "select") return "combobox";
    if (tag === "textarea") return "textbox";
    if (/^h[1-6]$/.test(tag)) return "heading";
    if (tag === "th") return "columnheader";
    if (tag === "td") return "cell";
    if (el.hasAttribute("onclick")) return "button";
    return "text";
  };

  // Legacy-table label: the text of the nearest non-empty cell to the LEFT of
  // the control's cell, in the same row.
  const tableLabel = (el) => {
    const td = el.closest("td,th");
    if (!td) return "";
    let p = td.previousElementSibling;
    while (p) {
      const t = norm(p.innerText);
      if (t) return t;
      p = p.previousElementSibling;
    }
    return "";
  };

  const labelFor = (el) => {
    if (el.id) {
      const l = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
      if (l) return norm(l.innerText);
    }
    const wl = el.closest("label");
    if (wl) return norm(wl.innerText);
    return "";
  };

  const nameOf = (el, role) => {
    const aria = el.getAttribute("aria-label");
    if (aria) return norm(aria);
    const tag = el.tagName.toLowerCase();
    if (tag === "input") {
      const t = (el.getAttribute("type") || "text").toLowerCase();
      if (["submit", "button", "reset"].includes(t)) return norm(el.value) || t;
      if (t === "image") return norm(el.alt) || "image";
      return labelFor(el) || norm(el.placeholder) || norm(el.title) || tableLabel(el) || norm(el.name);
    }
    if (tag === "select" || tag === "textarea") {
      return labelFor(el) || norm(el.title) || tableLabel(el) || norm(el.name);
    }
    if (tag === "a") {
      const img = el.querySelector("img[alt]");
      return norm(el.innerText) || (img ? norm(img.alt) : "") || norm(el.title) || norm(el.getAttribute("href"));
    }
    if (tag === "button") return norm(el.innerText) || norm(el.value);
    return norm(el.innerText);
  };

  const rowCtx = (el) => {
    const tr = el.closest("tr");
    if (!tr) return null;
    const table = tr.closest("table");
    if (!table) return null;
    const rows = Array.from(table.rows);
    const ri = rows.indexOf(tr);
    if (ri < 0) return null;
    const cellsOf = (r) => Array.from(r.cells).map((c) => norm(c.innerText).slice(0, 60));
    return { row_index: ri, headers: cellsOf(rows[0]), cells: cellsOf(tr), table_path: nthPath(table) };
  };

  const attrsOf = (el, role) => {
    const tag = el.tagName.toLowerCase();
    const a = {};
    const t = el.getAttribute("type");
    if (t) a.type = t.toLowerCase();
    if (el.getAttribute("name")) a.name = el.getAttribute("name");
    if (tag === "a" && el.getAttribute("href")) a.href = el.getAttribute("href");
    if (el.placeholder) a.placeholder = el.placeholder;
    if (tag === "input" || tag === "textarea") {
      const isPw = (a.type || "") === "password";
      const v = el.value || "";
      a.value = isPw ? (v ? "[masked]" : "") : v.slice(0, 40);
    }
    if (tag === "select") {
      a.options = Array.from(el.options).map((o) => norm(o.text)).slice(0, 20);
      a.value = el.value;
    }
    if (tag === "input" && (a.type === "checkbox" || a.type === "radio")) a.checked = !!el.checked;
    return a;
  };

  const out = [];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
  let el = root;
  do {
    const tag = el.tagName.toLowerCase();
    if (SKIP.has(tag) || INLINE.has(tag)) continue;
    if (!isVisible(el)) continue;
    const role = roleOf(el);
    if (role === null) continue;
    let name;
    if (INTERACTIVE.has(role)) {
      name = nameOf(el, role);
    } else {
      // Text-like element: emit only if it is the innermost block carrying
      // this text (no non-inline child has text of its own).
      const text = norm(el.innerText);
      if (!text) continue;
      if (hasBlockChildWithText(el)) continue;
      name = text;
    }
    const r = el.getBoundingClientRect();
    out.push({
      role: role,
      name: (name || "").slice(0, MAX_NAME),
      tag: tag,
      bbox: { x: r.left, y: r.top, w: r.width, h: r.height },
      structural: nthPath(el),
      row_ctx: rowCtx(el),
      attrs: attrsOf(el, role),
    });
  } while ((el = walker.nextNode()));

  return {
    elements: out,
    text: norm(root.innerText).slice(0, 4000),
    title: document.title || "",
    url: location.href,
  };
}
