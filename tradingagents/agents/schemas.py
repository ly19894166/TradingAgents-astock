"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context. Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used
- Schema field descriptions become the model's output instructions
- A render helper turns the parsed Pydantic instance back into markdown
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class T1Decision(str, Enum):
    """T+1 short-term decision used by the final Portfolio Manager."""

    BUY = "Buy"
    WAIT = "Wait"
    REJECT = "Reject"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    The Research Manager remains on the original five-tier research scale.
    The final Portfolio Manager later converts the research into a dedicated
    T+1 Buy / Wait / Reject decision.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )

    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )

    strategic_actions: str = Field(
        description=(
            "Concrete research-oriented steps for the trader to implement the "
            "recommendation, consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and trader context."""

    return "\n".join(
        [
            f"**Recommendation**: {plan.recommendation.value}",
            "",
            f"**Rationale**: {plan.rationale}",
            "",
            f"**Strategic Actions**: {plan.strategic_actions}",
        ]
    )


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The Trader reads the Research Manager's investment plan and analyst
    reports, then states a direction and reasoning.

    It deliberately carries no executable price levels.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )

    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences. Do not quote specific "
            "entry, stop-loss or position-size levels."
        ),
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown."""

    return "\n".join(
        [
            f"**Action**: {proposal.action.value}",
            "",
            f"**Reasoning**: {proposal.reasoning}",
            "",
            f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
        ]
    )


# ---------------------------------------------------------------------------
# Portfolio Manager — T+1 short-term decision
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured T+1 decision produced by the Portfolio Manager.

    The final Portfolio Manager is deliberately different from the upstream
    medium-term portfolio-rating model.

    Its job is to answer one question:

    Is the opportunity strong enough to take on trading day T when the primary
    monetisation / exit window is the next trading day's morning session?

    The final decision is therefore Buy / Wait / Reject rather than the
    upstream Buy / Overweight / Hold / Underweight / Sell scale.
    """

    rating: T1Decision = Field(
        description=(
            "Final T+1 short-term decision. Exactly one of Buy / Wait / Reject. "
            "Buy means the expected T+1 opportunity is strong enough to take on trading day T; "
            "Wait means DO NOT initiate a position on trading day T. The setup may remain "
            "on a watchlist for a future independent decision, but Wait authorizes no "
            "purchase under the current decision; Reject means the expected reward does "
            "not justify the overnight and next-day exit risk."
        ),
    )

    executive_summary: str = Field(
        description=(
            "Concise T+1 executive summary explaining why the setup is Buy, "
            "Wait, or Reject. Focus on the opportunity from the current decision "
            "point through the next trading day's morning session. Include the "
            "main positive driver, main risk, and overall asymmetry. Do not quote "
            "specific entry, stop-loss, position-size or target-price levels."
        ),
    )

    investment_thesis: str = Field(
        description=(
            "Detailed T+1 reasoning anchored in specific evidence from the "
            "analysts' reports, bull/bear debate, trader proposal, capital flow, "
            "news, policy, technical structure and risk debate. Give priority to "
            "information that can realistically affect price from trading day T "
            "through the next trading day's morning session. Longer-term facts "
            "may be used only when they affect this short-term window."
        ),
    )

    tomorrow_buyer: str = Field(
        description=(
            "Tomorrow Buyer: identify the most plausible source of incremental "
            "buying on the next trading day. Explain who may buy, why they may "
            "buy, and what evidence supports that view. Possible buyers may "
            "include trend followers, sector momentum capital, institutions, "
            "event-driven capital, short-term traders, passive/rebalancing flows "
            "or other identifiable participants. If no credible next-day buyer "
            "can be identified, state that clearly."
        ),
    )

    overnight_catalyst: str = Field(
        description=(
            "Overnight Catalyst: assess public catalysts that could matter "
            "between today's close and the next trading day's morning session. "
            "Distinguish confirmed catalysts from speculation. Consider company "
            "announcements, industry events, commodity moves, overseas markets, "
            "policy developments, earnings, orders, prices, supply-demand changes "
            "and other relevant public information. Explicitly state when no "
            "meaningful overnight catalyst exists."
        ),
    )

    price_in: str = Field(
        description=(
            "Price-in assessment. Start with exactly one of LOW / MEDIUM / HIGH, "
            "then explain how much of the positive thesis appears already reflected "
            "in the current price and recent price action. Consider recent gains, "
            "volume expansion, gap moves, sector crowding, technical extension and "
            "whether the catalyst was known before the latest move."
        ),
    )

    remaining_edge: str = Field(
        description=(
            "Remaining Edge: assess whether enough risk-adjusted upside remains "
            "from the current decision point through the next trading day's "
            "morning exit window. Explicitly classify the remaining edge as "
            "STRONG / MARGINAL / INSUFFICIENT and explain the evidence. A strong "
            "historical thesis is not enough if most of the move has already "
            "been priced in."
        ),
    )

    gap_scenarios: str = Field(
        description=(
            "T+1 opening scenarios. Separately assess HIGH OPEN / FLAT OPEN / "
            "LOW OPEN. For each scenario explain what it would imply about the "
            "overnight thesis, what evidence should be watched, and whether the "
            "preferred response is to continue holding briefly, seek an exit on "
            "strength, wait for confirmation, or exit early. Do not assume a gap "
            "direction without evidence."
        ),
    )

    auction_plan: str = Field(
        description=(
            "Opening call-auction plan for the next trading day, focused on "
            "09:15-09:25 Beijing time. Explain what auction strength, weakness, "
            "volume, relative strength and sector confirmation would confirm, "
            "weaken or invalidate the overnight thesis. Do not invent unavailable "
            "real-time auction data."
        ),
    )

    open_5_15_plan: str = Field(
        description=(
            "Next-day first 5-15 minute plan after 09:30. Explain what price "
            "behavior, volume, relative strength, sector behavior and follow-through "
            "would justify holding briefly versus exiting early. Explicitly consider "
            "failed strength, gap-and-fade behavior and weak rebound attempts."
        ),
    )

    exit_plan: str = Field(
        description=(
            "T+1 exit plan. The strategy is designed to monetise the overnight "
            "edge primarily during the next trading day's morning session rather "
            "than becoming a medium-term holding. Explain how to respond to: "
            "1) strong opening continuation, 2) early spike, 3) high-open fade, "
            "4) flat-open breakout or failure, 5) low-open repair, "
            "6) failed rebound, and 7) persistent weakness. Do not provide exact "
            "personalised executable price levels."
        ),
    )

    invalidation_conditions: str = Field(
        description=(
            "List the specific observable conditions that would invalidate the "
            "T+1 thesis and require abandoning the original bullish expectation "
            "rather than waiting for it to recover. Include relevant price-action, "
            "sector, catalyst, relative-strength, liquidity or risk signals. "
            "Do not invent data."
        ),
    )

    time_horizon: str = Field(
        default="T+1 next trading day morning",
        description=(
            "Use the T+1 short-term horizon only: decision on trading day T, "
            "overnight holding risk, and primary monetisation/exit window on the "
            "next trading day's morning session. Do not convert the analysis into "
            "a multi-week or multi-month investment thesis."
        ),
    )


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render the T+1 PortfolioDecision to markdown.

    The report intentionally exposes all T+1 decision fields so the user can
    audit why the final Buy / Wait / Reject conclusion was reached.
    """

    return "\n".join(
        [
            f"**Rating**: {decision.rating.value}",
            "",
            f"**Executive Summary**: {decision.executive_summary}",
            "",
            f"**Investment Thesis**: {decision.investment_thesis}",
            "",
            f"**Tomorrow Buyer**: {decision.tomorrow_buyer}",
            "",
            f"**Overnight Catalyst**: {decision.overnight_catalyst}",
            "",
            f"**Price-in**: {decision.price_in}",
            "",
            f"**Remaining Edge**: {decision.remaining_edge}",
            "",
            f"**T+1 Gap Scenarios**: {decision.gap_scenarios}",
            "",
            f"**Opening Auction Plan**: {decision.auction_plan}",
            "",
            f"**First 5-15 Minutes Plan**: {decision.open_5_15_plan}",
            "",
            f"**T+1 Exit Plan**: {decision.exit_plan}",
            "",
            f"**Invalidation Conditions**: {decision.invalidation_conditions}",
            "",
            f"**Time Horizon**: {decision.time_horizon}",
        ]
    )