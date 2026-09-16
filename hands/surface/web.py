"""Playwright implementation of the Surface protocol.

Key behaviours, all verified against a frameset page before this was written:
- perception runs the walker in EVERY frame and offsets boxes to page
  coordinates using frame_element().bounding_box() + clientLeft/Top;
- page-level locators never pierce frames, so actions always go through the
  owning Frame object (stable across that frame's navigations);
- one dialog listener is registered for the life of the page. In
  `auto_accept` mode (discovery) dialogs are answered immediately and
  recorded; in `hold` mode (replay / human control) they stay open and must
  be answered explicitly, and DOM calls are refused while one is open because
  the page is blocked;
- every mutating call presents a ControlToken to the guard.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from playwright.sync_api import Dialog, Frame, Locator, Page

from hands.schema.control import ControlToken
from hands.surface.base import Action, ActionResult, DialogInfo, Element, FrameInfo, Observation

_PERCEIVE_JS = (Path(__file__).parent / "perceive.js").read_text(encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class DialogOpen(RuntimeError):
    """A held dialog is open; the page is blocked until it is answered."""


class WebSurface:
    def __init__(
        self,
        page: Page,
        guard: Optional[Callable[[ControlToken], None]] = None,
        dialog_mode: str = "auto_accept",
        action_timeout_ms: int = 8000,
        hold_click_timeout_ms: int = 2500,
    ) -> None:
        self.page = page
        self._guard = guard
        self.dialog_mode = dialog_mode
        self.action_timeout_ms = action_timeout_ms
        # In "hold" mode a click that opens a dialog blocks the page's JS
        # thread, so the click call itself cannot return. We cap the wait and
        # then check whether a dialog arrived (see `act`).
        self.hold_click_timeout_ms = hold_click_timeout_ms
        self._pending: Optional[Dialog] = None
        self._pending_info: Optional[DialogInfo] = None
        self._dialogs_since_action: list[DialogInfo] = []
        self._idx_map: dict[int, tuple[Frame, str]] = {}
        self.last_observation: Optional[Observation] = None
        page.on("dialog", self._on_dialog)

    # ------------------------------------------------------------------ dialogs
    def _on_dialog(self, dialog: Dialog) -> None:
        info = DialogInfo(type=dialog.type, message=dialog.message, opened_at=_now())
        if self.dialog_mode == "auto_accept":
            try:
                dialog.accept()
                info.handled, info.accepted = True, True
            except Exception:  # noqa: BLE001 - dialog may already be gone
                pass
            self._dialogs_since_action.append(info)
        else:
            self._pending, self._pending_info = dialog, info

    def set_dialog_mode(self, mode: str) -> None:
        assert mode in ("auto_accept", "hold")
        self.dialog_mode = mode

    def dialog_state(self) -> Optional[DialogInfo]:
        if self._pending is None:
            return None
        # A human may have answered it natively (headed mode). Probe cheaply:
        # if the page responds, the dialog is gone and the driver entry is stale.
        try:
            self.page.wait_for_function("() => true", timeout=250)
        except Exception:  # noqa: BLE001
            return self._pending_info
        self._clear_stale_dialog()
        return None

    def _clear_stale_dialog(self) -> None:
        if self._pending is not None:
            try:
                self._pending.accept()
            except Exception:  # noqa: BLE001 - "No dialog is showing" is expected
                pass
            if self._pending_info is not None:
                self._pending_info.handled = True
                self._dialogs_since_action.append(self._pending_info)
        self._pending, self._pending_info = None, None

    def answer_dialog(self, accept: bool) -> Optional[DialogInfo]:
        if self._pending is None:
            return None
        info = self._pending_info
        try:
            if accept:
                self._pending.accept()
            else:
                self._pending.dismiss()
            if info:
                info.handled, info.accepted = True, accept
        except Exception:  # noqa: BLE001
            pass
        if info:
            self._dialogs_since_action.append(info)
        self._pending, self._pending_info = None, None
        return info

    def _refuse_if_blocked(self) -> None:
        if self.dialog_state() is not None:
            raise DialogOpen(f"{self._pending_info.type}: {self._pending_info.message}")  # type: ignore[union-attr]

    # --------------------------------------------------------------- perception
    def _frames(self) -> list[tuple[Frame, FrameInfo]]:
        frames = list(self.page.frames)
        out: list[tuple[Frame, FrameInfo]] = []
        for i, f in enumerate(frames):
            parent = f.parent_frame
            pidx = frames.index(parent) if parent in frames else None
            ox = oy = 0.0
            if parent is not None:
                try:
                    fe = f.frame_element()
                    bb = fe.bounding_box()
                    cl = fe.evaluate("e => [e.clientLeft || 0, e.clientTop || 0]")
                    if bb:
                        ox, oy = bb["x"] + cl[0], bb["y"] + cl[1]
                except Exception:  # noqa: BLE001 - detached frame
                    pass
            out.append((f, FrameInfo(index=i, name=f.name or "", url=f.url, parent_index=pidx, offset_x=ox, offset_y=oy)))
        return out

    def observe(self, screenshot_path: Optional[str] = None) -> Observation:
        dialog = self.dialog_state()
        elements: list[Element] = []
        frame_infos: list[FrameInfo] = []
        texts: dict[int, str] = {}
        titles: dict[int, str] = {}
        self._idx_map = {}
        if dialog is None:
            idx = 0
            for frame, info in self._frames():
                frame_infos.append(info)
                try:
                    res = frame.evaluate(_PERCEIVE_JS)
                except Exception as e:  # noqa: BLE001
                    texts[info.index] = f"[perception failed: {e.__class__.__name__}]"
                    continue
                texts[info.index] = res.get("text", "")
                titles[info.index] = res.get("title", "")
                for raw in res.get("elements", []):
                    idx += 1
                    bb = raw["bbox"]
                    el = Element(
                        idx=idx, role=raw["role"], name=raw["name"], tag=raw["tag"],
                        frame_index=info.index, frame_url=info.url, bbox=bb,
                        page_bbox={"x": bb["x"] + info.offset_x, "y": bb["y"] + info.offset_y, "w": bb["w"], "h": bb["h"]},
                        structural=raw["structural"], row_ctx=raw.get("row_ctx"), attrs=raw.get("attrs") or {},
                    )
                    elements.append(el)
                    self._idx_map[idx] = (frame, raw["structural"])
        shot = self.screenshot(screenshot_path) if screenshot_path else None
        obs = Observation(
            url=self.page.url, title=self._safe_title(), frames=frame_infos, elements=elements,
            frame_texts=texts, frame_titles=titles, dialog=dialog, screenshot_path=shot, ts=_now(),
        )
        self.last_observation = obs
        return obs

    def _safe_title(self) -> str:
        try:
            return self.page.title()
        except Exception:  # noqa: BLE001
            return ""

    def url(self) -> str:
        return self.page.url

    def frame_by_index(self, index: int) -> Frame:
        return list(self.page.frames)[index]

    # ------------------------------------------------------------------ evidence
    def screenshot(self, path: str) -> Optional[str]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        masks: list[Locator] = [f.locator("input[type=password]") for f in self.page.frames]
        try:
            self.page.screenshot(path=path, mask=masks, timeout=4000)
            return path
        except Exception:  # noqa: BLE001 - e.g. dialog open
            try:
                self.page.screenshot(path=path, timeout=1500)
                return path
            except Exception:  # noqa: BLE001
                return None

    def snapshot_dom(self, path: str) -> Optional[str]:
        if self.dialog_state() is not None:
            return None
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        parts = []
        for i, f in enumerate(self.page.frames):
            try:
                parts.append(f"<!-- frame {i} name={f.name!r} url={f.url} -->\n{f.content()}")
            except Exception as e:  # noqa: BLE001
                parts.append(f"<!-- frame {i} unavailable: {e.__class__.__name__} -->")
        Path(path).write_text("\n\n".join(parts), encoding="utf-8")
        return path

    # -------------------------------------------------------------------- acting
    def locator_for_idx(self, idx: int) -> Locator:
        if idx not in self._idx_map:
            raise KeyError(f"element {idx} is not in the last observation")
        frame, structural = self._idx_map[idx]
        loc = frame.locator(structural)
        if loc.count() != 1:
            raise LookupError(f"element {idx} is stale (structural path matched {loc.count()} nodes)")
        return loc

    def act(self, action: Action, token: ControlToken) -> ActionResult:
        if self._guard is not None:
            self._guard(token)
        self._dialogs_since_action = []
        t0 = time.perf_counter()
        try:
            self._refuse_if_blocked()
            self._perform(action)
            self._settle()
            ok, err = True, None
        except Exception as e:  # noqa: BLE001
            if self._pending is not None:
                # The action fired and opened a dialog we are deliberately
                # holding. The page is blocked, which is the only reason the
                # call could not return. That is a dialog, not a failure: the
                # caller decides what to do with it.
                ok, err = True, None
            else:
                ok, err = False, f"{e.__class__.__name__}: {str(e).splitlines()[0][:300]}"
        dialogs = list(self._dialogs_since_action)
        if self._pending_info is not None:
            dialogs.append(self._pending_info)
        return ActionResult(
            ok=ok, error=err, duration_ms=int((time.perf_counter() - t0) * 1000),
            url_after=self.page.url, dialogs=dialogs,
        )

    def _perform(self, a: Action) -> None:
        t = self.action_timeout_ms
        if a.kind == "navigate":
            assert a.url, "navigate needs a url"
            self.page.goto(a.url, wait_until="load", timeout=t)
            return
        if a.kind == "wait":
            self.page.wait_for_timeout(min(a.ms or 500, 10000))
            return
        if a.kind == "press" and a.element_idx is None and a.locator is None:
            self.page.keyboard.press(a.key or "Enter")
            return
        loc: Locator = a.locator if a.locator is not None else self.locator_for_idx(a.element_idx)  # type: ignore[arg-type]
        if a.kind == "click":
            click_t = self.hold_click_timeout_ms if self.dialog_mode == "hold" else t
            loc.click(timeout=click_t, no_wait_after=True)
        elif a.kind == "type":
            loc.fill(a.value or "", timeout=t)
        elif a.kind == "select":
            try:
                loc.select_option(value=a.value, timeout=t)
            except Exception:  # noqa: BLE001
                loc.select_option(label=a.value, timeout=t)
        elif a.kind == "press":
            loc.press(a.key or "Enter", timeout=t)
        else:
            raise ValueError(f"unknown action kind {a.kind}")

    def click_at(self, x: float, y: float) -> None:
        """Coordinate fallback: click a page-coordinate point (no DOM needed)."""
        self.page.mouse.click(x, y)

    def _settle(self) -> None:
        if self._pending is not None:
            return  # a held dialog blocks the page; waiting would only time out
        try:
            self.page.wait_for_load_state("load", timeout=3000)
        except Exception:  # noqa: BLE001
            pass
        for f in self.page.frames:
            try:
                f.wait_for_load_state("load", timeout=3000)
            except Exception:  # noqa: BLE001
                pass
        try:
            self.page.wait_for_timeout(150)
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        try:
            self.page.context.close()
        except Exception:  # noqa: BLE001
            pass
