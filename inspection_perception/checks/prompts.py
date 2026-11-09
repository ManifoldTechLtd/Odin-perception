"""把 `CheckItem` 拼成给 VLM 的 prompt。

小模型（~2B）的输出稳定性极度依赖 prompt 里的格式约束。同一个问题：
    "阀门关了吗？"                  → "The valve appears to be in a closed
                                       position, though it's hard to tell..."
    "…… Answer with only yes or no" → "yes"
后者才能被可靠解析。所以这里的核心工作就三件：
    1. 强制输出格式（只答 yes/no、只从选项里选、只输出读到的文本）
    2. 显式给出「不确定」的出口，让模型有路可退而不是硬猜
    3. 用英文写指令

第 3 点需要解释：Moondream2 这个量级的模型，指令遵循能力在英文上明显好于中文，
中文指令下经常无视格式约束、直接输出一段中文描述。所以**指令模板用英文，
但 `question` 字段本身允许中文**——业务侧写中文问题不影响格式约束生效。
"""
from __future__ import annotations

from ..schemas import CheckItem, CheckKind

# 「不确定」的统一出口。判定层认这个词，见 verdict.py。
UNCLEAR = "unclear"

_YES_NO_SUFFIX = (
    f"Answer with only one word: yes, no, or {UNCLEAR}. "
    f"Answer {UNCLEAR} if the image is too blurry, dark, or obstructed to tell."
)

_READ_TEXT_SUFFIX = (
    f"Answer with only the text or number you can read, nothing else. "
    f"Answer {UNCLEAR} if no text is legible."
)

_COUNT_SUFFIX = (
    f"Answer with only a single number. Answer {UNCLEAR} if you cannot count them reliably."
)


# 句末标点：半角/全角问号、半角/全角句号。
# 全角字符用转义写，避免编辑器或剪贴把全角问号静默转成半角后看不出来。
_SENTENCE_ENDINGS = ("?", "\uff1f", ".", "\u3002")


def build_prompt(item: CheckItem) -> str:
    """按检查类型给问题加上格式约束。"""
    question = item.question.strip()
    if not question:
        raise ValueError(f"CheckItem {item.id} 的 question 为空")
    if not question.endswith(_SENTENCE_ENDINGS):
        question += "?"

    if item.kind is CheckKind.YES_NO:
        return f"{question} {_YES_NO_SUFFIX}"

    if item.kind is CheckKind.STATE:
        if not item.options:
            raise ValueError(f"CheckItem {item.id} 是 STATE 类型但没给 options")
        allowed = ", ".join(item.options)
        return (
            f"{question} Answer with exactly one of these words: {allowed}, {UNCLEAR}. "
            f"Answer {UNCLEAR} if none of them clearly applies."
        )

    if item.kind is CheckKind.READ_TEXT:
        return f"{question} {_READ_TEXT_SUFFIX}"

    if item.kind is CheckKind.COUNT:
        return f"{question} {_COUNT_SUFFIX}"

    if item.kind is CheckKind.DESCRIBE:
        # 自由描述不加约束，本来就是要它多说
        return question

    raise ValueError(f"未支持的 CheckKind: {item.kind}")
