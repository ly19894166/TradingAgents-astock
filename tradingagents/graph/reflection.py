# TradingAgents/graph/reflection.py

from typing import Any


class Reflector:
    """Handles reflection on T+1 trading decisions."""

    def __init__(self, quick_thinking_llm: Any):
        """Initialize the reflector with an LLM."""
        self.quick_thinking_llm = quick_thinking_llm
        self.log_reflection_prompt = self._get_log_reflection_prompt()

    def _get_log_reflection_prompt(self) -> str:
        """Reflection prompt aligned with the T+1 short-term system."""
        return (
            "You are reviewing a past A-share T+1 short-term decision after "
            "the next trading day's opening outcome is known.\n"
            "Write exactly 2-4 sentences of plain prose "
            "(no bullets, no headers, no markdown).\n\n"

            "OUTCOME WINDOW:\n"
            "The recorded return is T close -> T+1 open. "
            "It is a deterministic proxy for the overnight / opening edge "
            "because the deferred logger currently uses daily OHLC data. "
            "It is NOT a five-day return and is NOT claimed to be the exact "
            "first-5-to-15-minute realised exit price.\n\n"

            "DECISION SEMANTICS:\n"
            "- Buy means taking the T-day T+1 opportunity was justified. "
            "Positive T+1-open alpha supports the call; materially negative "
            "alpha weakens it.\n"
            "- Wait means NO position was authorised on trading day T. "
            "Do not call Wait incorrect merely because price later rose. "
            "A strong positive T+1-open move may indicate a missed opportunity; "
            "flat, weak or negative outcomes generally support waiting.\n"
            "- Reject also means NO T-day position. "
            "Strong positive realised edge may indicate an overly conservative "
            "rejection; weak or negative edge generally supports rejection.\n\n"

            "Cover in order:\n"
            "1. Was the Buy / Wait / Reject decision appropriate for this "
            "T+1 window? Use alpha as outcome evidence.\n"
            "2. Which part of Tomorrow Buyer, Overnight Catalyst, Price-in, "
            "Remaining Edge or the risk thesis held or failed?\n"
            "3. Give one concrete lesson for the next similar T+1 decision.\n\n"

            "Do not reinterpret the result as a swing trade or medium-term "
            "investment outcome."
        )

    def reflect_on_final_decision(
        self,
        final_decision: str,
        raw_return: float,
        alpha_return: float,
    ) -> str:
        """Reflect on a final T+1 decision using the T-close -> T+1-open proxy."""
        messages = [
            ("system", self.log_reflection_prompt),
            (
                "human",
                (
                    f"T close -> T+1 open raw return: {raw_return:+.1%}\n"
                    f"T close -> T+1 open alpha vs CSI 300 "
                    f"(沪深300): {alpha_return:+.1%}\n\n"
                    f"Final T+1 Decision:\n{final_decision}"
                ),
            ),
        ]

        return self.quick_thinking_llm.invoke(messages).content
