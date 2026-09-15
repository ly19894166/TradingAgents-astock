"""Extract the final T+1 decision from the Portfolio Manager output."""

from __future__ import annotations

from typing import Any

from tradingagents.agents.utils.rating import parse_t1_decision


class SignalProcessor:
    """Read the final Buy / Wait / Reject decision."""

    def __init__(self, quick_thinking_llm: Any = None):
        # Kept only for backwards compatibility.
        # No extra LLM call is required.
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """Return exactly one of Buy / Wait / Reject."""
        return parse_t1_decision(full_signal)