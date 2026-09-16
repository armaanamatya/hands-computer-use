"""Deterministic replay: run a saved `Capability` against a live `Surface`
with no LLM in the decision loop. See `hands.replay.engine` for the
per-step decision procedure."""
from __future__ import annotations

from hands.replay.engine import HumanResolution, ReplayEngine

__all__ = ["ReplayEngine", "HumanResolution"]
