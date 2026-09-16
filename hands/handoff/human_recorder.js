// Human-action recorder. Injected into every frame when control passes to a
// human operator, so the run's evidence records what the person actually did
// on the live session - not just that a handoff happened.
//
// Two injection paths are needed and both are used by controller.py:
//   - frame.evaluate(this) for documents that are ALREADY loaded
//     (add_init_script only affects documents loaded afterwards);
//   - context.add_init_script(this) for documents loaded later.
// Hence the idempotence guard: this may run twice in the same document.
//
// Values are NOT captured by default. The binding receives a descriptor of
// the control plus a flag saying whether a value exists; the Python side
// decides, from policy.capture_human_input_values, whether to keep it. A
// password field never reports a value at all.
() => {
  if (window.__handsRecorderInstalled) return "already-installed";
  if (typeof window.hands_record !== "function") return "no-binding";
  window.__handsRecorderInstalled = true;

  const norm = (s) => (s || "").replace(/\s+/g, " ").trim();

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

  const roleOf = (el) => {
    const tag = el.tagName.toLowerCase();
    if (tag === "a") return "link";
    if (tag === "button") return "button";
    if (tag === "select") return "combobox";
    if (tag === "textarea") return "textbox";
    if (tag === "input") {
      const t = (el.getAttribute("type") || "text").toLowerCase();
      if (["submit", "button", "reset", "image"].includes(t)) return "button";
      if (t === "checkbox") return "checkbox";
      if (t === "radio") return "radio";
      return "textbox";
    }
    return tag;
  };

  // Same label heuristics as the perception walker, so a human-performed step
  // can be described the same way an agent-performed one is.
  const labelOf = (el) => {
    const aria = el.getAttribute && el.getAttribute("aria-label");
    if (aria) return norm(aria);
    if (el.id) {
      const l = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
      if (l) return norm(l.innerText);
    }
    const wrap = el.closest && el.closest("label");
    if (wrap) return norm(wrap.innerText);
    const td = el.closest && el.closest("td,th");
    if (td) {
      let p = td.previousElementSibling;
      while (p) {
        const t = norm(p.innerText);
        if (t) return t;
        p = p.previousElementSibling;
      }
    }
    if (el.value && ["submit", "button", "reset"].includes((el.getAttribute("type") || "").toLowerCase())) {
      return norm(el.value);
    }
    return norm(el.innerText) || norm(el.name) || "";
  };

  const describe = (el) => {
    if (!el || el.nodeType !== 1) return null;
    const type = (el.getAttribute && el.getAttribute("type") || "").toLowerCase();
    return {
      role: roleOf(el),
      name: labelOf(el).slice(0, 120),
      tag: el.tagName.toLowerCase(),
      structural: nthPath(el),
      attrs: {
        type: type || undefined,
        name: (el.getAttribute && el.getAttribute("name")) || undefined,
        href: (el.tagName.toLowerCase() === "a" && el.getAttribute("href")) || undefined,
      },
    };
  };

  const send = (payload) => {
    try {
      window.hands_record(Object.assign({ frame_url: location.href, ts: Date.now() }, payload));
    } catch (e) { /* binding torn down with the handoff; nothing to do */ }
  };

  document.addEventListener("click", (ev) => {
    const el = ev.target && ev.target.closest
      ? ev.target.closest("a,button,input,select,textarea,td,th") || ev.target
      : ev.target;
    send({ type: "click", target: describe(el) });
  }, true);

  document.addEventListener("change", (ev) => {
    const el = ev.target;
    if (!el || !el.tagName) return;
    const type = (el.getAttribute("type") || "").toLowerCase();
    const isPassword = type === "password";
    const d = describe(el);
    const payload = { type: el.tagName.toLowerCase() === "select" ? "select" : "input", target: d };
    // has_value lets the operator log show that something was entered even
    // when the value itself is withheld.
    payload.has_value = !isPassword && !!el.value;
    payload.sensitive = isPassword;
    if (!isPassword) payload.value = (el.value || "").slice(0, 120);
    if (type === "checkbox" || type === "radio") payload.checked = !!el.checked;
    send(payload);
  }, true);

  document.addEventListener("keydown", (ev) => {
    // Only the keys that commit or cancel a form are interesting; logging
    // every keystroke would be both noisy and a data-handling liability.
    if (ev.key === "Enter" || ev.key === "Escape") {
      send({ type: "keypress", key: ev.key, target: describe(ev.target) });
    }
  }, true);

  window.addEventListener("submit", (ev) => {
    send({ type: "submit", target: describe(ev.target) });
  }, true);

  return "installed";
}
