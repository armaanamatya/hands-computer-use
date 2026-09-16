"""Surface abstraction: how the system perceives and acts on a UI.

The agent loop and the replay engine only ever talk to a `Surface`. The web
implementation (Playwright) is the one built here; the shape of `Observation`
and `Element` is deliberately the seam for other surfaces: a desktop backend
would produce the same element list from UI Automation / Accessibility APIs
(role, name, bbox) and act by coordinates or automation patterns.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Protocol

from hands.schema.control import ControlToken

ActionKind = Literal["navigate", "click", "type", "select", "press", "wait"]


@dataclass
class FrameInfo:
    index: int
    name: str
    url: str
    parent_index: Optional[int]
    offset_x: float = 0.0   # page-coordinate offset of this frame's viewport origin
    offset_y: float = 0.0


@dataclass
class Element:
    idx: int
    role: str
    name: str
    tag: str
    frame_index: int
    frame_url: str
    bbox: dict[str, float]          # frame-relative
    page_bbox: dict[str, float]     # page coordinates (bbox + frame offset)
    structural: str
    row_ctx: Optional[dict[str, Any]] = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def is_interactive(self) -> bool:
        return self.role in {"link", "button", "textbox", "checkbox", "radio", "combobox"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DialogInfo:
    type: str
    message: str
    opened_at: str
    handled: bool = False
    accepted: Optional[bool] = None


@dataclass
class Observation:
    url: str
    title: str
    frames: list[FrameInfo]
    elements: list[Element]
    frame_texts: dict[int, str]          # frame_index -> visible text excerpt
    frame_titles: dict[int, str]
    dialog: Optional[DialogInfo]
    screenshot_path: Optional[str]
    ts: str

    def element(self, idx: int) -> Element:
        for e in self.elements:
            if e.idx == idx:
                return e
        raise KeyError(f"no element with idx {idx} in the last observation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "frames": [asdict(f) for f in self.frames],
            "elements": [e.to_dict() for e in self.elements],
            "frame_texts": {str(k): v for k, v in self.frame_texts.items()},
            "frame_titles": {str(k): v for k, v in self.frame_titles.items()},
            "dialog": asdict(self.dialog) if self.dialog else None,
            "screenshot_path": self.screenshot_path,
            "ts": self.ts,
        }


@dataclass
class Action:
    kind: ActionKind
    element_idx: Optional[int] = None   # discovery: index into the last observation
    value: Optional[str] = None         # type / select
    key: Optional[str] = None           # press
    url: Optional[str] = None           # navigate
    ms: Optional[int] = None            # wait
    locator: Any = None                 # replay: an already-resolved Playwright Locator
    reason: str = ""


@dataclass
class ActionResult:
    ok: bool
    error: Optional[str] = None
    duration_ms: int = 0
    url_after: Optional[str] = None
    dialogs: list[DialogInfo] = field(default_factory=list)


class Surface(Protocol):
    def observe(self, screenshot_path: Optional[str] = None) -> Observation: ...
    def act(self, action: Action, token: ControlToken) -> ActionResult: ...
    def screenshot(self, path: str) -> Optional[str]: ...
    def snapshot_dom(self, path: str) -> Optional[str]: ...
    def url(self) -> str: ...
    def dialog_state(self) -> Optional[DialogInfo]: ...
    def close(self) -> None: ...
