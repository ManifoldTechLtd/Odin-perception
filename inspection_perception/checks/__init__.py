"""检查逻辑层：prompt 拼装、答案判定，以及「定位 + 确定性算子」路径。

    build_prompt(item)          CheckItem → 带格式约束的 prompt（自由问答路径）
    judge(item, raw_answer)     模型原始回答 → (status, 规范化答案, 说明)
    answer_check(rgb, item, be) 一个入口：有 rule 走算子，否则走自由问答
    validate_rule(rule, id)     加载清单时校 rule 合法性
    classify_region(rgb, ...)   区域主色判定（指示灯类算子的核心）

verdict/prompts 是纯函数，可脱离 GPU 与 ROS 单测；ops/color 也只依赖 numpy，
用合成图即可验证——巡检判定的正确性主要靠这一层的测试来守。
"""
from __future__ import annotations

from .color import classify_region
from .ops import OpOutcome, RuleError, answer_check, describe_ops, validate_rule
from .prompts import UNCLEAR, build_prompt
from .verdict import (
    judge,
    looks_uncertain,
    parse_count,
    parse_state,
    parse_text,
    parse_yes_no,
)

__all__ = [
    "UNCLEAR",
    "OpOutcome",
    "RuleError",
    "answer_check",
    "build_prompt",
    "classify_region",
    "describe_ops",
    "judge",
    "looks_uncertain",
    "parse_yes_no",
    "parse_state",
    "parse_text",
    "parse_count",
    "validate_rule",
]
