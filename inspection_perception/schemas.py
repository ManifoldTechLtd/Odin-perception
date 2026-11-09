"""巡检检查的数据契约。

这层先定下来，是因为它决定了上下游能不能各自独立演进：
    - 上游（巡检导航/任务调度）产出 `CheckItem`：到了哪个点位、要查什么
    - 下游（VLM 后端）只回答自然语言问题，不关心业务
    - 中间的判定规则把「模型说了什么」翻译成「合格/不合格」
换模型（Moondream2 → Qwen2.5-VL）只影响 `vlm/`，换判定策略只影响 `checks/`，
两者都不动 `CheckItem` / `CheckResult`，也不动调度侧。

所有 dataclass 都能 `to_dict()` 成纯 JSON 类型，方便落盘存证与走 HTTP。
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class CheckKind(str, Enum):
    """检查类型。对应巡检里最常见的几种「看一眼就能判」的活儿。

    先只实现前三种，够覆盖绝大多数点位；后两种留接口。
    继承 str 是为了 `json.dumps` 能直接序列化。
    """

    YES_NO = "yes_no"        # 是非判断：阀门是否关闭、有无渗漏、门是否关好
    STATE = "state"          # 多选一状态：指示灯 红/绿/灭、开关 开/关
    READ_TEXT = "read_text"  # 读文字/数字：铭牌、编号、数显表
    COUNT = "count"          # 计数：几个消防器材（待实现）
    DESCRIBE = "describe"    # 自由描述：兜底记录异常（待实现）


class CheckStatus(str, Enum):
    """检查结论。

    `UNKNOWN` 是刻意保留的第三态——巡检场景里「看不清所以不敢判」必须能表达出来，
    强行二分类会把模型的犹豫变成假阴性，比漏报更危险。
    """

    PASS = "pass"        # 符合预期
    FAIL = "fail"        # 不符合预期，需要人介入
    UNKNOWN = "unknown"  # 无法判定（画面模糊、遮挡、模型答非所问）
    ERROR = "error"      # 流程出错（取不到图、模型加载失败）


# 归一化 ROI：(x0, y0, x1, y1)，取值 0~1。
# 用归一化而非像素，是为了同一份任务配置能在不同分辨率/缩放下复用。
Roi = Tuple[float, float, float, float]


@dataclass
class CheckItem:
    """一条检查任务：在某个点位要确认的一件事。"""

    id: str
    """稳定的唯一标识，用于关联历史记录做时序对比。"""

    question: str
    """喂给 VLM 的问题，用自然语言。写得越具体，结果越稳。"""

    kind: CheckKind = CheckKind.YES_NO

    name: str = ""
    """人类可读的名字，用于报表。留空则回落到 id。"""

    expect: Optional[str] = None
    """期望答案。
    YES_NO   : "yes" 或 "no"
    STATE    : options 中的某一项，如 "green"
    READ_TEXT: 留空表示只记录不判定；填正则表示需匹配
    """

    options: List[str] = field(default_factory=list)
    """STATE 类型的候选状态。会拼进 prompt 约束模型只从中选。"""

    target: str = ""
    """检测式后端要找的目标短语（英文，如 "oil leak" / "green indicator light"）。

    只有 Florence-2 这类**不能自由问答**、只吃固定任务标签的模型会用到它：
    它们把「是否有渗漏」实现为「开词表检测 'oil leak'，有框就是 yes」。
    Moondream2 / SmolVLM 这类自由问答模型完全忽略这个字段。

    留空时检测式后端会回落到拿 `question` 当短语，但准确率明显下降——
    一整句问话当检测词条用本来就不对。所以要跑 profile B 就请把它填上。
    """

    roi: Optional[Roi] = None
    """只看画面的这块区域。表盘/指示灯这类小目标必须裁，否则细节被缩放吃掉。"""

    rule: Optional[Dict[str, Any]] = None
    """用确定性算子而不是自由问答来得出答案。结构与参数见 `checks/ops.py`。

    例子（阀门手柄是否竖直）::

        rule:
          op: segment_orientation
          target: "valve handle"
          map: {vertical: "yes", horizontal: "no"}

    为什么需要它：巡检里很多「是非/状态」判断本质上不是语义问题，而是几何或颜色
    问题（手柄角度、指示灯色相）。交给模型用语言回答，就把一个确定性问题变成了
    概率性问题；改成「模型只负责定位，算子负责判定」后，同一张图永远同一个结论，
    判错了还能从存证里看到是框错了还是阈值不对。

    留空则走自由问答路径，老清单不用改。
    类型用 dict 而不是 dataclass：各算子的参数完全不同，硬套一个平铺的 dataclass
    会得到十几个互斥的 Optional 字段。合法性由 `checks.ops.validate_rule()` 在
    加载清单时校，拼错参数名会当场报错而不是静默忽略。
    """

    location: str = ""
    """点位描述，例如 "1号泵房-东侧"。仅用于报表，不参与推理。"""

    severity: str = "normal"
    """不合格时的严重程度: low | normal | high。供上层决定是否立即告警。"""

    def display_name(self) -> str:
        return self.name or self.id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckItem":
        data = dict(data)
        if "kind" in data:
            data["kind"] = CheckKind(data["kind"])
        if data.get("roi") is not None:
            data["roi"] = tuple(float(v) for v in data["roi"])  # YAML 读出来是 list
        _reject_unquoted_yaml_bools(data)
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"CheckItem 有未知字段 {sorted(unknown)}（id={data.get('id')}）")
        return cls(**data)


def _reject_unquoted_yaml_bools(data: Dict[str, Any]) -> None:
    """拦住 YAML 1.1 的 bool 词陷阱。

    `options: [on, off]` / `expect: yes` 不加引号时，PyYAML 会把 on/off/yes/no
    解析成 True/False。这类错在加载期不暴露，而是溜到判定层炸成
    「decoding to str: need a bytes-like object, bool found」这种和配置毫不相干的
    TypeError——在这里提前拦住，并直接告诉用户怎么改。
    """
    item_id = data.get("id", "")
    expect = data.get("expect")
    if expect is not None and not isinstance(expect, str):
        raise ValueError(
            f"expect 必须是字符串，收到 {expect!r}（id={item_id}）——"
            f"yes/no/on/off 不加引号会被 YAML 解析成 bool，请给 expect 加引号"
        )
    bad = [v for v in (data.get("options") or []) if not isinstance(v, str)]
    if bad:
        raise ValueError(
            f"options 里有非字符串项 {bad}（id={item_id}）——"
            f"on/off/yes/no 不加引号会被 YAML 解析成 bool，请给每个选项加引号"
        )


@dataclass
class Evidence:
    """判定依据。出了争议要能复现，所以原图和原始回答都留着。"""

    image_path: str = ""
    """存证图片路径（已裁 ROI 的那张，即模型实际看到的画面）。"""

    raw_answer: str = ""
    """VLM 的原始输出，未经解析。"""

    prompt: str = ""
    """实际发给模型的完整 prompt，含 ROI 与选项约束。

    算子路径下这里是真正发出去的任务标签串（如
    `<REFERRING_EXPRESSION_SEGMENTATION>valve handle`），不是自然语言问句。
    """

    detail: Dict[str, Any] = field(default_factory=dict)
    """算子的结构化中间结果：检测框、主轴角度、颜色票数。

    这是算子路径比自由问答多出来的可观测性，也是它真正的价值所在。
    模型回一句 "no" 时你无法知道它为什么说 no；而这里能看到
    「检测到 1 个框但占整图 0.97 被面积上限滤掉」——判断对不对、
    阈值该不该改，看一眼就清楚。
    """

    model: str = ""
    """模型标识，如 "moondream2@2025-04-14"。换模型后回溯要用。"""

    frame_stamp_ns: int = 0
    """图像的 ROS 时间戳，可与机器人位姿/日志对齐。"""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CheckResult:
    """一条检查的结论。"""

    item_id: str
    status: CheckStatus
    answer: str = ""
    """规范化后的答案：yes/no、green、读到的文本。"""

    reason: str = ""
    """给人看的说明，特别是 UNKNOWN/ERROR 时的原因。"""

    latency_ms: float = 0.0
    """端到端耗时，含取图+推理。用于评估 NX 上的实际吞吐。"""

    evidence: Evidence = field(default_factory=Evidence)
    checked_at: float = field(default_factory=time.time)

    @property
    def ok(self) -> bool:
        return self.status is CheckStatus.PASS

    @property
    def needs_attention(self) -> bool:
        """是否需要人看一眼。UNKNOWN 也算——宁可多看，不可漏判。"""
        return self.status in (CheckStatus.FAIL, CheckStatus.UNKNOWN, CheckStatus.ERROR)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    def summary_line(self) -> str:
        mark = {
            CheckStatus.PASS: "OK  ",
            CheckStatus.FAIL: "FAIL",
            CheckStatus.UNKNOWN: "?   ",
            CheckStatus.ERROR: "ERR ",
        }[self.status]
        extra = f"  ({self.reason})" if self.reason else ""
        return f"[{mark}] {self.item_id}: {self.answer}{extra}  {self.latency_ms:.0f}ms"
