"""把 VLM 的自由文本回答判成 PASS / FAIL / UNKNOWN。

这一层是整个方案最容易出错、也最需要防守的地方。小模型即便被要求「只答 yes 或 no」，
实际输出仍会是 "Yes." / "yes, it is closed" / "The valve is closed." / "I can't tell"。
解析必须对这些变体都稳，并且**宁可判 UNKNOWN 也不要猜**。

三条防守原则
------------
**① 用词边界匹配，不用子串匹配。** `"no" in "not visible"` 是 True，
`"no" in "unknown"` 也是 True——子串匹配会把一堆回答误判成 "no"。全部走 `\b` 正则。

**② 先查「不确定」再查答案。** "I cannot tell if it is closed" 里既有 hedge 词
也有 "closed"，必须让 hedge 优先，否则会把模型的犹豫读成肯定答案。

**③ yes 和 no 同时出现时判 UNKNOWN。** 例如 "yes, but no leak is visible"，
两个都命中说明解析不可靠，交给人看。
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..schemas import CheckItem, CheckKind, CheckStatus
from .prompts import UNCLEAR

# 模型表达「我不确定」的常见说法。命中任一即判 UNKNOWN。
_HEDGE_PATTERNS = [
    r"\bunclear\b",
    r"\bunknown\b",
    # Florence-2 对答不出的输入（如无文字的 ROI 做 OCR）固定返回 "unanswerable"，
    # 与 cannot tell 同类——不列为 hedge 词的话，读数类检查会把「没读到」误判成
    # 「读数不合格」（FAIL），而它应当是需要人看一眼的 UNKNOWN。
    r"\bunanswerable\b",
    r"\bnot sure\b",
    r"\bnot clear\b",
    r"\bcannot (?:tell|determine|see|read)\b",
    r"\bcan'?t (?:tell|determine|see|read)\b",
    r"\bunable to\b",
    r"\bdifficult to (?:tell|determine|see)\b",
    r"\bhard to (?:tell|determine|see)\b",
    r"\bno(?:t)? (?:enough|sufficient) (?:detail|information)\b",
    r"\btoo (?:blurry|dark|small)\b",
    r"不确定",
    r"看不清",
    r"无法(?:判断|确定|识别)",
]
_HEDGE_RE = re.compile("|".join(_HEDGE_PATTERNS), re.IGNORECASE)

_YES_RE = re.compile(r"\b(?:yes|yeah|yep|true|correct)\b|是的|正确", re.IGNORECASE)
# 中文否定词不能用单字「否」：它是「是否」的子串，
# 模型回一句「是否关闭不确定」就会被误判成否定。只用双字以上的否定词。
_NO_RE = re.compile(
    r"\b(?:no|nope|false|incorrect|negative)\b|不是|没有|不能|否定", re.IGNORECASE
)

# 模型爱加的前缀，读文本时要剥掉
_TEXT_PREFIX_RE = re.compile(
    r"^(?:the (?:text|number|label|sign|reading)\s*(?:reads|says|is)?\s*:?\s*"
    r"|it (?:reads|says)\s*:?\s*"
    r"|answer\s*:?\s*)",
    re.IGNORECASE,
)


def looks_uncertain(answer: str) -> bool:
    """回答是否表达了不确定。"""
    return bool(_HEDGE_RE.search(answer))


def parse_yes_no(answer: str) -> Optional[str]:
    """解析是非题。返回 "yes" / "no" / None(无法判定)。"""
    if looks_uncertain(answer):
        return None
    has_yes = bool(_YES_RE.search(answer))
    has_no = bool(_NO_RE.search(answer))
    if has_yes and has_no:
        return None  # 自相矛盾，不猜
    if has_yes:
        return "yes"
    if has_no:
        return "no"
    return None


def parse_state(answer: str, options: List[str]) -> Optional[str]:
    """解析多选一。命中零个或多于一个都返回 None。"""
    if looks_uncertain(answer):
        return None
    hits = [
        opt for opt in options
        if re.search(rf"\b{re.escape(opt)}\b", answer, re.IGNORECASE)
        or (not opt.isascii() and opt in answer)  # 中文选项没有词边界，退化成子串
    ]
    return hits[0] if len(hits) == 1 else None


# 引号（半角双/单、中文弯引号）与句末句号，全角字符用转义写以免被静默替换。
_STRIP_QUOTES = "\"'\u201c\u201d"
_STRIP_PERIODS = ".\u3002"


def parse_text(answer: str) -> Optional[str]:
    """解析读数/读字，剥掉模型爱加的前缀和引号。"""
    if looks_uncertain(answer):
        return None
    text = _TEXT_PREFIX_RE.sub("", answer.strip())
    text = text.strip().strip(_STRIP_QUOTES).rstrip(_STRIP_PERIODS).strip()
    return text or None


def parse_count(answer: str) -> Optional[int]:
    """解析计数。取回答里第一个整数。"""
    if looks_uncertain(answer):
        return None
    match = re.search(r"\b(\d+)\b", answer)
    return int(match.group(1)) if match else None


def judge(item: CheckItem, raw_answer: str) -> Tuple[CheckStatus, str, str]:
    """把原始回答判成结论。

    Returns:
        (status, 规范化答案, 说明)。说明只在非 PASS 时有内容。
    """
    raw = (raw_answer or "").strip()
    if not raw:
        return CheckStatus.UNKNOWN, "", "模型返回空"

    if item.kind is CheckKind.YES_NO:
        value = parse_yes_no(raw)
        if value is None:
            return CheckStatus.UNKNOWN, "", f"无法从回答中解析出 yes/no: {raw[:80]!r}"
        expect = (item.expect or "yes").strip().lower()
        if expect not in ("yes", "no"):
            return CheckStatus.ERROR, value, f"expect 应为 yes/no，配置里是 {item.expect!r}"
        if value == expect:
            return CheckStatus.PASS, value, ""
        return CheckStatus.FAIL, value, f"期望 {expect}，实际 {value}"

    if item.kind is CheckKind.STATE:
        value = parse_state(raw, item.options)
        if value is None:
            return (
                CheckStatus.UNKNOWN,
                "",
                f"回答未命中唯一选项 {item.options}: {raw[:80]!r}",
            )
        if not item.expect:
            # 没给期望值就只做记录，不判合格与否
            return CheckStatus.PASS, value, ""
        if value.lower() == item.expect.strip().lower():
            return CheckStatus.PASS, value, ""
        return CheckStatus.FAIL, value, f"期望 {item.expect}，实际 {value}"

    if item.kind is CheckKind.READ_TEXT:
        value = parse_text(raw)
        if value is None:
            return CheckStatus.UNKNOWN, "", f"未读出有效文本: {raw[:80]!r}"
        if not item.expect:
            return CheckStatus.PASS, value, ""
        # expect 当正则用，能同时表达精确匹配和范围（如 r"^0\.[3-6]\d*$"）
        try:
            matched = re.search(item.expect, value) is not None
        except re.error as exc:
            return CheckStatus.ERROR, value, f"expect 不是合法正则 {item.expect!r}: {exc}"
        if matched:
            return CheckStatus.PASS, value, ""
        return CheckStatus.FAIL, value, f"读到 {value!r}，不匹配 {item.expect!r}"

    if item.kind is CheckKind.COUNT:
        value = parse_count(raw)
        if value is None:
            return CheckStatus.UNKNOWN, "", f"未解析出数量: {raw[:80]!r}"
        if not item.expect:
            return CheckStatus.PASS, str(value), ""
        try:
            expected = int(item.expect)
        except ValueError:
            return CheckStatus.ERROR, str(value), f"expect 应为整数，是 {item.expect!r}"
        if value == expected:
            return CheckStatus.PASS, str(value), ""
        return CheckStatus.FAIL, str(value), f"期望 {expected} 个，数到 {value} 个"

    if item.kind is CheckKind.DESCRIBE:
        # 自由描述只记录，永不判 FAIL——它的价值是留痕给人看
        return CheckStatus.PASS, raw, ""

    return CheckStatus.ERROR, "", f"未支持的 CheckKind: {item.kind}"


__all__ = [
    "UNCLEAR",
    "judge",
    "looks_uncertain",
    "parse_yes_no",
    "parse_state",
    "parse_text",
    "parse_count",
]
