"""VLM 后端接口。

两个协议，对应两类能力：
    VLMBackend       ask/caption          看图回答问题（自由问答模型）
    GroundingBackend detect/segment/ocr   定位与读字（Florence-2 这类专用模型）

一个后端可以只实现前者（Moondream2、SmolVLM）、也可以两者都实现（Florence-2）。
`checks/ops.py` 会按后端实际有哪些方法选执行路径，缺能力时明确报错而不是静默降级。

为什么后来还是加了 `GroundingBackend`
------------------------------------
这层接口最初刻意只有 ask/caption，理由是「Moondream2 有 point()、Qwen2.5-VL 有
grounding、VLM-FO1 有 region token，三家表达不同，硬抽象只会得到谁都不好用的
最小公倍数」。这个顾虑本身没错，但前提变了：确定用 Florence-2 做端侧主力之后，
**它不能自由问答，能力全在定位上**——不把定位抽进接口，就只能让业务层去 import
具体后端类，那才是真的耦合。

抽象的粒度也因此收窄成三个「所有定位型模型都能表达」的动作：给框、给轮廓、读字。
没有 region token、没有 dense caption、没有 grounding score——那些各家确实不通用，
等真有需求再谈。

新增后端只需实现协议，然后在 `registry.py` 里注册一行。

关于 `hint`
-----------
`ask()` 多了一个可选的 `hint`。加它是因为 Florence-2 这类模型**根本不能自由
问答**——它只认 `<OCR>`、`<OPEN_VOCABULARY_DETECTION>` 这样的固定任务标签，
拿到一句自然语言问句会答非所问。要让它接进同一条流水线，后端必须知道
「这条检查是什么类型、要找什么目标」，而这个信息在 `question` 字符串里已经丢了。

没有把它做成必填参数，也没有让 `ask()` 直接收 `CheckItem`：
前者会逼所有自由问答后端改签名，后者会把业务数据结构漏进模型层。
`AskHint` 是一个只含推理所需字段的窄接口，自由问答后端收下后直接忽略即可。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from ..geometry import Box


@dataclass
class AskHint:
    """给后端的结构化提示。自由问答后端可以完全忽略。

    刻意用 `str` 而不是 `CheckKind`：`vlm/` 不该 import `schemas`，
    否则模型层就跟业务数据结构绑死了。
    """

    kind: str = ""
    """检查类型，取 `CheckKind` 的字符串值：yes_no / state / read_text / count / describe。"""

    target: str = ""
    """检测式后端要找的英文短语。见 `CheckItem.target`。"""

    options: List[str] = field(default_factory=list)
    """state 类型的候选状态。"""


@runtime_checkable
class GroundingBackend(Protocol):
    """定位型后端：能给框、给轮廓、读字，但不一定能自由问答。

    这三个方法返回**结构化结果**而不是文本，因为下游要拿它们算几何
    （见 `geometry.py` 与 `checks/ops.py`）。把框转成 "yes"/"no" 这种
    语义压缩发生在 `checks/ops.py`，不在后端里做——后端只负责「看见了什么」，
    「看见这个算不算合格」是业务判断。
    """

    def detect(self, rgb: np.ndarray, phrase: str) -> List[Box]:
        """开词表检测：找 `phrase` 描述的目标，返回像素坐标的框。

        找不到时返回空列表，**不要抛异常**——「画面里没有渗漏」是正常且重要的结论。
        """
        ...

    def segment(self, rgb: np.ndarray, phrase: str) -> List[List[Tuple[float, float]]]:
        """指代分割：返回若干多边形轮廓，每个是 [(x, y), ...] 像素坐标。

        比 `detect` 贵一点，但轴对齐外接框判不出斜置目标的姿态，
        「阀门手柄是否竖直」这类检查必须走轮廓。
        """
        ...

    def ocr(self, rgb: np.ndarray) -> str:
        """读出画面里的文字。读不到返回空字符串。"""
        ...


@runtime_checkable
class VLMBackend(Protocol):
    """视觉语言模型后端。实现方无需显式继承，结构匹配即可。"""

    name: str
    """模型标识，会写进 Evidence 用于结果回溯，建议带版本，如 "moondream2@2025-04-14"。"""

    def ask(
        self,
        rgb: np.ndarray,
        question: str,
        max_new_tokens: int = 64,
        hint: Optional[AskHint] = None,
    ) -> str:
        """看图回答问题。

        Args:
            rgb: HxWx3 的 uint8 RGB 数组。
            question: 自然语言问题。
            max_new_tokens: 输出长度上限。是非题给小值能显著提速。
            hint: 结构化提示。自由问答后端忽略；只吃任务标签的后端（Florence-2）
                依赖它决定用哪个任务标签，缺失时只能降级成整图描述。

        Returns:
            模型的原始文本输出（不做解析，解析在 `checks/` 层做）。
            检测式后端会把检测结果**合成**成 yes/no、数字或文本，
            这样 `checks/verdict.py` 对所有后端都是同一套解析逻辑。
        """
        ...

    def caption(self, rgb: np.ndarray, length: str = "short") -> str:
        """描述整张图。length: "short" | "normal"。"""
        ...

    def warmup(self) -> None:
        """跑一次空推理，把 CUDA kernel 和显存分配的一次性开销提前付掉。

        不做这步的话，第一条检查的耗时会比后续高好几倍，
        在 NX 上尤其明显，会污染性能测量结果。
        """
        ...

    def unload(self) -> None:
        """释放模型显存。NX 内存紧张，跑完一轮巡检可以主动放掉。"""
        ...


class VLMError(RuntimeError):
    """后端加载或推理失败。

    单独一个异常类型，是为了让 pipeline 能把「模型的问题」判成
    `CheckStatus.ERROR` 而不是 `FAIL`——前者是系统故障要报修，
    后者是设备异常要处置，两者不能混。
    """


def check_rgb(rgb: Optional[np.ndarray]) -> np.ndarray:
    """校验输入图像。各后端复用，把维度问题挡在推理之前。"""
    if rgb is None:
        raise VLMError("输入图像为 None")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise VLMError(f"需要 HxWx3 的 RGB 图，收到 shape={rgb.shape}")
    if rgb.dtype != np.uint8:
        raise VLMError(f"需要 uint8，收到 dtype={rgb.dtype}")
    return rgb
