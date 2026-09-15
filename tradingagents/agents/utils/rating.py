"""Shared rating and T+1 decision parsers.

Two vocabularies intentionally coexist in this fork:

1. Research-stage 5-tier rating:
   Buy / Overweight / Hold / Underweight / Sell

2. Final T+1 decision:
   Buy / Wait / Reject

The Research Manager keeps the original 5-tier vocabulary.
The final Portfolio Manager uses the dedicated T+1 vocabulary.

Keeping the parsers separate prevents a final Wait / Reject decision from
silently degrading to Hold while preserving compatibility with the upstream
research stages.
"""

from __future__ import annotations

import re
from typing import Tuple


# ---------------------------------------------------------------------------
# Original 5-tier research rating
# ---------------------------------------------------------------------------

RATINGS_5_TIER: Tuple[str, ...] = (
    "Buy",
    "Overweight",
    "Hold",
    "Underweight",
    "Sell",
)

_RATING_SET = {r.lower() for r in RATINGS_5_TIER}

_RATING_LABEL_RE = re.compile(
    r"rating.*?[:\-][\s*]*(\w+)",
    re.IGNORECASE,
)

_CN_RATING_MAP = {
    "强烈买入": "Buy",
    "买入": "Buy",
    "买进": "Buy",
    "增持": "Overweight",
    "持有": "Hold",
    "中性": "Hold",
    "观望": "Hold",
    "维持": "Hold",
    "减持": "Underweight",
    "强烈卖出": "Sell",
    "清仓": "Sell",
    "卖出": "Sell",
}

_CN_ALT = "|".join(
    sorted(_CN_RATING_MAP, key=len, reverse=True)
)

_CN_LABEL_PREFIX = (
    r"(?:最终评级|评级|投资评级|评级结论|最终投资建议|投资建议|操作建议|"
    r"推荐评级|建议|推荐)\s*[:：\-]\s*\*{0,2}\s*"
)

_CN_LABEL_RE = re.compile(
    _CN_LABEL_PREFIX + r"(" + _CN_ALT + r")"
)

_WORD_CONTINUATION = r"(?:[A-Za-z0-9_]|-(?=[A-Za-z0-9_]))"

_RATING_VALUE_END = (
    r"(?!\*{0,2}" + _WORD_CONTINUATION + r")"
)

_CN_LABEL_EN_RE = re.compile(
    _CN_LABEL_PREFIX
    + r"("
    + "|".join(RATINGS_5_TIER)
    + r")"
    + _RATING_VALUE_END,
    re.IGNORECASE,
)

_CN_TERM_RE = re.compile(_CN_ALT)


def parse_rating(text: str, default: str = "Hold") -> str:
    """Extract the original 5-tier research rating.

    Returns one of:

        Buy
        Overweight
        Hold
        Underweight
        Sell

    This function is kept for compatibility with upstream research-stage code.
    """

    if not text:
        return default

    # 1. Explicit English Rating label.
    for line in text.splitlines():
        match = _RATING_LABEL_RE.search(line)
        if match and match.group(1).lower() in _RATING_SET:
            return match.group(1).capitalize()

    # 2. Explicit Chinese label + Chinese rating.
    match = _CN_LABEL_RE.search(text)
    if match:
        return _CN_RATING_MAP[match.group(1)]

    # 3. Explicit Chinese label + English rating.
    match = _CN_LABEL_EN_RE.search(text)
    if match:
        return match.group(1).capitalize()

    # 4. Bare English research rating.
    for line in text.splitlines():
        for word in line.lower().split():
            clean = word.strip("*:.,;()[]{}，。；：（）【】")
            if clean in _RATING_SET:
                return clean.capitalize()

    # 5. Bare Chinese research rating.
    match = _CN_TERM_RE.search(text)
    if match:
        return _CN_RATING_MAP[match.group(0)]

    return default


# ---------------------------------------------------------------------------
# Final T+1 decision
# ---------------------------------------------------------------------------

T1_DECISIONS: Tuple[str, ...] = (
    "Buy",
    "Wait",
    "Reject",
)

_T1_SET = {d.lower() for d in T1_DECISIONS}

# Strong preference for explicitly labelled final-decision fields.
_T1_EN_LABEL_RE = re.compile(
    r"(?:rating|final\s+decision|t\+1\s+decision|decision)"
    r".*?[:\-]\s*\*{0,2}\s*"
    r"(Buy|Wait|Reject)"
    r"(?!\*{0,2}(?:[A-Za-z0-9_]|-(?=[A-Za-z0-9_])))",
    re.IGNORECASE,
)

_T1_CN_MAP = {
    "买入": "Buy",
    "可买": "Buy",
    "可以买": "Buy",
    "等待": "Wait",
    "继续等待": "Wait",
    "观望": "Wait",
    "暂缓": "Wait",
    "拒绝": "Reject",
    "放弃": "Reject",
    "取消": "Reject",
    "不买": "Reject",
    "避免": "Reject",
}

_T1_CN_ALT = "|".join(
    sorted(_T1_CN_MAP, key=len, reverse=True)
)

_T1_CN_LABEL_PREFIX = (
    r"(?:最终评级|最终决策|T\+1决策|T1决策|短线决策|"
    r"交易决策|操作决策|决策|评级)"
    r"\s*[:：\-]\s*\*{0,2}\s*"
)

_T1_CN_LABEL_CN_RE = re.compile(
    _T1_CN_LABEL_PREFIX
    + r"("
    + _T1_CN_ALT
    + r")",
    re.IGNORECASE,
)

_T1_CN_LABEL_EN_RE = re.compile(
    _T1_CN_LABEL_PREFIX
    + r"(Buy|Wait|Reject)"
    + r"(?!\*{0,2}(?:[A-Za-z0-9_]|-(?=[A-Za-z0-9_])))",
    re.IGNORECASE,
)

# Exact markdown form emitted by render_pm_decision().
_T1_MARKDOWN_RATING_RE = re.compile(
    r"^\s*\*{0,2}Rating\*{0,2}"
    r"\s*[:：\-]\s*\*{0,2}"
    r"(Buy|Wait|Reject)"
    r"\*{0,2}\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_t1_decision(
    text: str,
    default: str = "Wait",
) -> str:
    """Extract the final T+1 Buy / Wait / Reject decision.

    Explicit labels are deliberately preferred over bare words because the
    report body may contain sentences discussing hypothetical Buy, Wait or
    Reject outcomes.

    Returns exactly one of:

        Buy
        Wait
        Reject
    """

    if not text:
        return default

    # 1. Exact markdown produced by our PortfolioDecision renderer.
    match = _T1_MARKDOWN_RATING_RE.search(text)
    if match:
        return match.group(1).capitalize()

    # 2. Other explicitly labelled English decision.
    match = _T1_EN_LABEL_RE.search(text)
    if match:
        return match.group(1).capitalize()

    # 3. Explicit Chinese label + English enum.
    match = _T1_CN_LABEL_EN_RE.search(text)
    if match:
        return match.group(1).capitalize()

    # 4. Explicit Chinese label + Chinese decision.
    match = _T1_CN_LABEL_CN_RE.search(text)
    if match:
        return _T1_CN_MAP[match.group(1)]

    # 5. Conservative fallback:
    #    only accept a line whose entire meaningful content is the decision.
    for line in text.splitlines():
        cleaned = line.strip().strip("*# `:：-")
        if cleaned.lower() in _T1_SET:
            return cleaned.capitalize()

        if cleaned in _T1_CN_MAP:
            return _T1_CN_MAP[cleaned]

    # Failure to parse a final T+1 decision should NOT silently become Buy.
    # Wait is the safe neutral fallback.
    return default