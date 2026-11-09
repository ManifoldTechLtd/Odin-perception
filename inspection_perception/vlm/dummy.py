"""桩后端：不加载任何模型，按预设规则返回答案。

存在的意义不是「占位」，而是让整条链路（取图 → ROI 裁剪 → prompt 拼装 →
答案解析 → 判定 → 存证）可以在没有 GPU、没有模型权重、没有 ROS 的机器上被完整测试。
CI 里也用它，跑测试不需要 4GB 权重。

它对 prompt 里的问题做朴素关键词匹配，因此能构造出确定性的 PASS / FAIL /
UNKNOWN 三种走向，用来验证判定逻辑本身是否正确。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .base import AskHint, check_rgb


class DummyBackend:
    """可编程的假 VLM。

    Args:
        answers: 问题关键词 → 固定答案。命中即返回，用于构造确定性测试。
        default: 没命中任何关键词时的回答。
    """

    def __init__(
        self,
        answers: Optional[Dict[str, str]] = None,
        default: str = "I am not sure.",
    ) -> None:
        self.name = "dummy@v1"
        self._answers = answers or {}
        self._default = default
        self.call_count = 0
        self.last_hint: Optional[AskHint] = None
        """最后一次收到的 hint。测试靠它确认流水线真的把检查类型传到了后端，
        否则 Florence-2 这类后端会默默降级成整图描述而没人发现。"""

    def ask(
        self,
        rgb: np.ndarray,
        question: str,
        max_new_tokens: int = 64,
        hint: Optional[AskHint] = None,
    ) -> str:
        check_rgb(rgb)
        self.call_count += 1
        self.last_hint = hint
        lowered = question.lower()
        for keyword, answer in self._answers.items():
            if keyword.lower() in lowered:
                return answer
        return self._default

    def caption(self, rgb: np.ndarray, length: str = "short") -> str:
        rgb = check_rgb(rgb)
        self.call_count += 1
        # 回一句包含真实尺寸和均值亮度的描述，这样测试能确认图确实传进来了
        return f"A {rgb.shape[1]}x{rgb.shape[0]} image with mean brightness {rgb.mean():.1f}."

    def warmup(self) -> None:
        pass

    def unload(self) -> None:
        pass
