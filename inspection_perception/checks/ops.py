"""检查算子：把「模型看见了什么」变成「合格与否的答案」。

这一层是 Florence-2 能不能用在巡检上的关键。

背景
----
Florence-2 不能自由问答，但它能框、能分割、能 OCR。巡检清单里大量检查项
其实并不需要语义推理，只需要"定位 + 一次确定性计算"：

    阀门手柄是否关闭   ->  分割手柄，算主轴角度，看是否接近竖直
    指示灯是什么颜色   ->  取 ROI 内最亮的像素，统计色相
    有几个消防器材     ->  开词表检测，去重，数框
    有没有渗漏        ->  开词表检测，按面积过滤掉整图幻觉框
    铭牌编号是多少     ->  OCR

把这些交给模型"用语言回答"，是让它做最不擅长的事。改成算子之后：
  - **确定性**：同一张图永远同一个结论，不会两次巡检给出相反答案
  - **可解释**：判错了能看到框在哪、角度多少、票数多少，指出是框错还是阈值不对
  - **可离线验证**：算子能用合成图证明正确性，不必等现场图（见 tests/test_ops.py）
  - **更快**：颜色类从 N 次推理降到 1 次

设计
----
清单里给一条 `rule`，声明用哪个算子、参数是什么：

    - id: valve_01_closed
      kind: yes_no
      expect: yes
      rule:
        op: segment_orientation
        target: "valve handle"
        map: {vertical: "yes", horizontal: "no"}

算子的返回值刻意做成和自由问答模型同构的字符串（"yes"/"no"/"green"/"3"/文本），
所以 `verdict.judge()` 对两条路径是同一套解析逻辑，一行都不用改。
不确定时返回 "unclear (原因)"，`judge` 会命中 hedge 词判 UNKNOWN——
**算子宁可判 UNKNOWN 也不猜**，和自由问答路径的原则一致。

没写 `rule` 的检查项走原来的自由问答路径，老清单不用改。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..geometry import (
    AMBIGUOUS,
    Box,
    HORIZONTAL,
    VERTICAL,
    angle_to_orientation,
    dedupe,
    filter_boxes,
    largest,
    orientation_from_box,
    polygon_area,
    polygon_bbox,
    polygon_orientation,
)
from ..schemas import CheckItem
from .color import GRAY, OFF, UNCLEAR as COLOR_UNCLEAR, WHITE, classify_region, resolve_palette
from .prompts import build_prompt

# 颜色判定里自动产生、不参与色相投票的名字。清单的 options 里写它们是合理的
# （"off" 是一种真实状态），但不能把它们当色带传给 classify_region。
AUTO_COLOR_NAMES = frozenset({OFF, WHITE, GRAY, COLOR_UNCLEAR})


class RuleError(RuntimeError):
    """算子无法执行：后端缺少所需能力，或 rule 与 item 对不上。

    单独一个类型，是为了让流水线能把它落成 `CheckStatus.ERROR`（系统问题）
    而不是 `FAIL`（设备问题）——"这个后端不会分割"不是设备不合格。
    """


@dataclass
class OpOutcome:
    """算子的产出。字段与 `Evidence` 对齐，流水线直接搬进存证。"""

    answer: str
    """给 `judge()` 解析的规范化答案。不确定时形如 "unclear (原因)"。"""

    raw: str = ""
    """原始依据的文本化摘要。自由问答路径就是模型原话；算子路径是算出来的中间量。"""

    prompt: str = ""
    """真正发给模型的东西。算子路径记的是任务标签串或算子调用描述。"""

    detail: Dict[str, Any] = field(default_factory=dict)
    """结构化中间结果（框、角度、票数）。判错了靠这个定位原因。"""


# =============================================================================
# 参数表。每个算子声明自己认识哪些键，多写一个键就在**加载清单时**报错。
#
# 不做这个校验的话，`max_area_frac` 手滑写成 `max_area_fraction` 会被静默忽略，
# 而它恰好是压制 Florence-2 幻觉框的关键参数——现场表现是"偶尔莫名误报"，
# 排查成本极高。宁可加载时就失败。
# =============================================================================
_COMMON_KEYS = {"op", "target", "mode"}
_OP_KEYS: Dict[str, set] = {
    "presence": _COMMON_KEYS | {"min_area_frac", "max_area_frac", "min_count",
                                "label_contains", "iou_thr"},
    "count": _COMMON_KEYS | {"min_area_frac", "max_area_frac", "label_contains", "iou_thr"},
    "box_orientation": _COMMON_KEYS | {"ratio", "map", "min_area_frac", "max_area_frac"},
    "segment_orientation": _COMMON_KEYS | {"tol_deg", "min_elongation", "map", "angle_map"},
    "region_color": _COMMON_KEYS | {"colors", "color_map", "locate", "bright_percentile",
                                    "min_frac", "off_v", "gray_s", "white_v",
                                    "min_area_frac", "max_area_frac"},
    "ocr_text": {"op"},
    "vqa": {"op"},
}

# 需要后端具备定位能力的算子。跑 Moondream2/SmolVLM 时清单里带这些 op 会明确报错，
# 而不是悄悄给出一个没有依据的答案。
_NEEDS_DETECT = {"presence", "count", "box_orientation", "region_color"}
_NEEDS_SEGMENT = {"segment_orientation"}


def validate_rule(rule: Optional[Dict[str, Any]], item_id: str = "") -> None:
    """校验 rule 的结构。加载清单时调用，配置错就别等到跑一半才发现。"""
    if rule is None:
        return
    where = f"（id={item_id}）" if item_id else ""
    if not isinstance(rule, dict):
        raise ValueError(f"rule 必须是字典，收到 {type(rule).__name__}{where}")

    op = rule.get("op")
    if op not in _OP_KEYS:
        raise ValueError(f"未知 rule.op: {op!r}{where}，可选 {sorted(_OP_KEYS)}")

    unknown = set(rule) - _OP_KEYS[op]
    if unknown:
        raise ValueError(
            f"rule.op={op} 不认识参数 {sorted(unknown)}{where}，"
            f"可用 {sorted(_OP_KEYS[op])}"
        )

    if op in ("box_orientation", "segment_orientation"):
        if not rule.get("map") and not rule.get("angle_map"):
            raise ValueError(
                f"rule.op={op} 必须给 map 或 angle_map{where}——"
                f"算子只知道\"竖直/水平\"，不知道竖直对应\"关闭\"还是\"打开\""
            )
        if rule.get("map") is not None:
            if not isinstance(rule["map"], dict):
                raise ValueError(f"rule.map 必须是字典{where}")
            # 算子只会吐出这两个方向（第三态 ambiguous 不靠 map 翻）。
            # 不校这一步的话，"verticle" 这种拼错会让所有结果变成 unclear，
            # 而日志里只能看到"map 里没有 vertical"，很难联想到是配置拼写错。
            bad = {str(k).lower() for k in rule["map"]} - {VERTICAL, HORIZONTAL}
            if bad:
                raise ValueError(
                    f"rule.map 的键只能是 {VERTICAL!r}/{HORIZONTAL!r}，"
                    f"收到 {sorted(bad)}{where}"
                )
            # 值也会踩 YAML 的 bool 词坑：map: {vertical: yes} 裸写 → True，
            # 算子把 True 当答案吐给判定层，同样炸成看不懂的 TypeError。
            bad_vals = sorted(k for k, v in rule["map"].items() if not isinstance(v, str))
            if bad_vals:
                raise ValueError(
                    f"rule.map 的值必须是字符串（{bad_vals} 的值不是）{where}——"
                    f"yes/no/on/off 不加引号会被 YAML 解析成 bool，请加引号"
                )
        for entry in rule.get("angle_map") or []:
            missing = {"from", "to", "answer"} - set(entry or {})
            if missing:
                raise ValueError(f"angle_map 条目缺 {sorted(missing)}{where}")
            if not isinstance(entry["answer"], str):
                raise ValueError(
                    f"angle_map 的 answer 必须是字符串，收到 {entry['answer']!r}{where}——"
                    f"on/off/yes/no 不加引号会被 YAML 解析成 bool，请加引号"
                )

    if op == "region_color" and rule.get("colors"):
        # 这里会对未知颜色名抛错，正是想要的：加载时就发现拼写错误
        resolve_palette([c for c in rule["colors"] if str(c).lower() not in AUTO_COLOR_NAMES])


# =============================================================================
# 入口
# =============================================================================
def answer_check(rgb: np.ndarray, item: CheckItem, backend: Any) -> OpOutcome:
    """取得一条检查的答案。有 `rule` 走算子，否则走自由问答。

    流水线只调这一个函数，两条路径的差异全部收在本模块里。
    """
    if item.rule:
        return _run_rule(rgb, item, backend)
    return _run_vqa(rgb, item, backend)


def _run_vqa(rgb: np.ndarray, item: CheckItem, backend: Any) -> OpOutcome:
    """自由问答路径：拼 prompt、带 hint 问一次。"""
    from ..vlm.base import AskHint

    prompt = build_prompt(item)
    hint = AskHint(kind=item.kind.value, target=item.target, options=list(item.options))
    raw = backend.ask(rgb, prompt, max_new_tokens=_token_budget(item), hint=hint)
    # 定位型后端不会把这句自然语言真的发出去，存证要记它实际发的任务标签串
    effective = getattr(backend, "last_prompt", "") or prompt
    return OpOutcome(answer=raw, raw=raw, prompt=effective, detail={"path": "vqa"})


def _token_budget(item: CheckItem) -> int:
    """按类型给输出长度上限。是非题给小值能明显降延迟。"""
    from ..schemas import CheckKind

    return {
        CheckKind.YES_NO: 8,
        CheckKind.STATE: 12,
        CheckKind.READ_TEXT: 32,
        CheckKind.COUNT: 8,
        CheckKind.DESCRIBE: 128,
    }.get(item.kind, 64)


def _run_rule(rgb: np.ndarray, item: CheckItem, backend: Any) -> OpOutcome:
    rule = dict(item.rule or {})
    op = rule.get("op", "")
    handler = _HANDLERS.get(op)
    if handler is None:  # validate_rule 应该已经挡住了，这里是双保险
        raise RuleError(f"未知 rule.op: {op!r}（id={item.id}）")

    backend_name = getattr(backend, "name", type(backend).__name__)
    if op in _NEEDS_DETECT and not callable(getattr(backend, "detect", None)):
        raise RuleError(
            f"rule.op={op} 需要后端支持 detect()，但 {backend_name} 没有。"
            f"这条检查请改用 op=vqa，或换 florence2 后端。"
        )
    if op in _NEEDS_SEGMENT and not callable(getattr(backend, "segment", None)):
        raise RuleError(
            f"rule.op={op} 需要后端支持 segment()，但 {backend_name} 没有。"
            f"退一步可用 op=box_orientation（只需 detect，但判不出斜置目标）。"
        )

    outcome = handler(rgb, item, backend, rule)
    outcome.detail.setdefault("op", op)
    if not outcome.prompt:
        outcome.prompt = getattr(backend, "last_prompt", "") or f"[op:{op}]"
    return outcome


# =============================================================================
# 各算子
# =============================================================================
def _phrase(item: CheckItem, rule: Dict[str, Any]) -> str:
    """检测短语：rule.target 优先，回落到 item.target。"""
    phrase = str(rule.get("target") or item.target or "").strip()
    if not phrase:
        raise RuleError(
            f"rule.op={rule.get('op')} 需要检测短语，但 rule.target 和 "
            f"item.target 都是空的（id={item.id}）"
        )
    return phrase


def _detect(rgb, backend, phrase: str, rule: Dict[str, Any]) -> List[Box]:
    mode = str(rule.get("mode") or "ovd")
    try:
        return list(backend.detect(rgb, phrase, mode=mode) or [])
    except TypeError:
        # 后端的 detect 可能没有 mode 参数（协议里 mode 是 Florence-2 的扩展）
        return list(backend.detect(rgb, phrase) or [])


def _kept_boxes(
    rgb: np.ndarray,
    backend: Any,
    phrase: str,
    rule: Dict[str, Any],
    default_max_area: float = 1.0,
) -> tuple:
    """检测 + 过滤 + 去重，返回 (保留的框, 全部框)。"""
    boxes = _detect(rgb, backend, phrase, rule)
    h, w = rgb.shape[:2]
    kept = filter_boxes(
        boxes,
        w,
        h,
        min_area_frac=float(rule.get("min_area_frac", 0.0)),
        max_area_frac=float(rule.get("max_area_frac", default_max_area)),
        label_contains=str(rule.get("label_contains", "")),
    )
    kept = dedupe(kept, iou_thr=float(rule.get("iou_thr", 0.6)))
    return kept, boxes


def _op_presence(rgb, item, backend, rule) -> OpOutcome:
    """有没有这个东西 -> yes / no。

    `max_area_frac` 默认取 0.9 而不是 1.0，这是本算子最重要的一个默认值：
    Florence-2 找不到目标时最典型的行为是**返回一个几乎覆盖整图的框**。
    不卡上限，"期望没有渗漏"这类检查会稳定误报 FAIL——而且看起来像是模型
    "确实检测到了渗漏"，非常容易被当成真问题去排查设备。
    """
    phrase = _phrase(item, rule)
    kept, boxes = _kept_boxes(rgb, backend, phrase, rule, default_max_area=0.9)
    min_count = int(rule.get("min_count", 1))
    answer = "yes" if len(kept) >= min_count else "no"
    detail = {
        "phrase": phrase,
        "boxes": [b.to_dict() for b in kept],
        "detected": len(boxes),
        "kept": len(kept),
        "min_count": min_count,
    }
    raw = f"detect({phrase!r}) -> {len(boxes)} 框，过滤后 {len(kept)} 框 -> {answer}"
    return OpOutcome(answer=answer, raw=raw, detail=detail)


def _op_count(rgb, item, backend, rule) -> OpOutcome:
    """数个数。去重是必须的，见 `geometry.dedupe` 的说明。"""
    phrase = _phrase(item, rule)
    kept, boxes = _kept_boxes(rgb, backend, phrase, rule)
    detail = {
        "phrase": phrase,
        "boxes": [b.to_dict() for b in kept],
        "detected": len(boxes),
        "deduped": len(kept),
    }
    raw = f"detect({phrase!r}) -> {len(boxes)} 框，去重后 {len(kept)}"
    return OpOutcome(answer=str(len(kept)), raw=raw, detail=detail)


def _map_orientation(orientation: str, rule: Dict[str, Any], extra: str) -> str:
    """把 vertical/horizontal 翻成业务答案。映射里没有就是 unclear。"""
    mapping = {str(k).lower(): str(v) for k, v in (rule.get("map") or {}).items()}
    if orientation == AMBIGUOUS:
        return f"unclear (姿态分辨不出: {extra})"
    answer = mapping.get(orientation)
    if answer is None:
        return f"unclear (map 里没有 {orientation}: {extra})"
    return answer


def _op_box_orientation(rgb, item, backend, rule) -> OpOutcome:
    """只凭外接框长宽比判姿态。便宜（1 次推理），但判不出斜置目标。

    斜 45° 的手柄外接框是正方形，本算子会诚实地返回 unclear 而不是猜一个。
    要处理斜置就用 `segment_orientation`。
    """
    phrase = _phrase(item, rule)
    kept, boxes = _kept_boxes(rgb, backend, phrase, rule)
    box = largest(kept)
    if box is None:
        return OpOutcome(
            answer=f"unclear (没检测到 {phrase!r})",
            raw=f"detect({phrase!r}) -> 0 框",
            detail={"phrase": phrase, "detected": len(boxes)},
        )

    ratio = float(rule.get("ratio", 1.3))
    orientation = orientation_from_box(box, ratio=ratio)
    wh = box.w / box.h if box.h > 0 else 0.0
    answer = _map_orientation(orientation, rule, f"w/h={wh:.2f} 阈值={ratio}")
    detail = {
        "phrase": phrase,
        "box": box.to_dict(),
        "w_over_h": round(wh, 3),
        "ratio_threshold": ratio,
        "orientation": orientation,
    }
    raw = f"box w/h={wh:.2f} -> {orientation} -> {answer}"
    return OpOutcome(answer=answer, raw=raw, detail=detail)


def _angle_bucket(angle: float, angle_map: Sequence[Dict[str, Any]]) -> Optional[str]:
    """按角度区间取答案。区间是无向角 [0,180)，`from > to` 表示跨 0° 环绕。

    环绕必须支持：竖直是 90° 不跨界，但"接近水平"是 [150,180)∪[0,30)，
    写成两条区间会让配置变啰嗦且容易漏一半。
    """
    a = angle % 180.0
    for entry in angle_map:
        lo = float(entry["from"]) % 180.0
        hi = float(entry["to"]) % 180.0
        inside = (lo <= a < hi) if lo <= hi else (a >= lo or a < hi)
        if inside:
            return str(entry["answer"])
    return None


def _op_segment_orientation(rgb, item, backend, rule) -> OpOutcome:
    """分割轮廓 -> 主轴角度 -> 业务答案。姿态类检查的首选。

    比 `box_orientation` 多一次分割的开销，但能处理任意角度，
    而且 `elongation` 顺带把"这东西根本不是杆状"这种情况挡掉了。
    """
    phrase = _phrase(item, rule)
    rings = list(backend.segment(rgb, phrase) or [])
    if not rings:
        return OpOutcome(
            answer=f"unclear (没分割出 {phrase!r})",
            raw=f"segment({phrase!r}) -> 0 轮廓",
            detail={"phrase": phrase, "rings": 0},
        )

    # 取面积最大的环：分割常带出反光、遮挡造成的碎片，拿碎片算主轴会完全跑偏
    ring = max(rings, key=polygon_area)
    angle, elongation = polygon_orientation(ring)

    tol_deg = float(rule.get("tol_deg", 30.0))
    min_elongation = float(rule.get("min_elongation", 1.25))
    detail = {
        "phrase": phrase,
        "rings": len(rings),
        "bbox": polygon_bbox(ring, phrase).to_dict(),
        "angle_deg": round(angle, 1),
        "elongation": round(elongation, 2),
    }

    angle_map = rule.get("angle_map")
    if angle_map:
        if elongation < min_elongation:
            answer = (
                f"unclear (目标不够细长 elongation={elongation:.2f}"
                f"<{min_elongation}，主轴方向不可信)"
            )
        else:
            hit = _angle_bucket(angle, angle_map)
            answer = hit or f"unclear (角度 {angle:.1f}° 不在任何 angle_map 区间内)"
        detail["min_elongation"] = min_elongation
    else:
        orientation = angle_to_orientation(
            angle, elongation, tol_deg=tol_deg, min_elongation=min_elongation
        )
        detail["orientation"] = orientation
        detail["tol_deg"] = tol_deg
        detail["min_elongation"] = min_elongation
        answer = _map_orientation(
            orientation, rule,
            f"angle={angle:.1f}° elongation={elongation:.2f} tol={tol_deg}°",
        )

    raw = f"segment({phrase!r}) -> angle={angle:.1f}° elongation={elongation:.2f} -> {answer}"
    return OpOutcome(answer=answer, raw=raw, detail=detail)


def _op_region_color(rgb, item, backend, rule) -> OpOutcome:
    """区域主色 -> 状态。指示灯类的首选，1 次推理甚至 0 次。

    `locate` 默认 False：清单里给指示灯配了 ROI 的话，ROI 就已经框住目标了，
    再让模型定位一次纯属多花一次推理，还多一个幻觉框的风险。
    只有目标在画面里会移动、没法用固定 ROI 时才开 locate。
    """
    colors = rule.get("colors") or item.options
    colors = [str(c) for c in colors if str(c).strip().lower() not in AUTO_COLOR_NAMES]

    box: Optional[Box] = None
    detail: Dict[str, Any] = {}
    if rule.get("locate"):
        phrase = _phrase(item, rule)
        kept, boxes = _kept_boxes(rgb, backend, phrase, rule)
        box = largest(kept)
        detail["phrase"] = phrase
        detail["detected"] = len(boxes)
        detail["box"] = box.to_dict() if box else None
        if box is None:
            return OpOutcome(
                answer=f"unclear (locate 开启但没检测到 {phrase!r})",
                raw=f"detect({phrase!r}) -> 0 框",
                detail=detail,
            )

    verdict = classify_region(
        rgb,
        box=box,
        colors=colors or None,
        bright_percentile=float(rule.get("bright_percentile", 75.0)),
        min_frac=float(rule.get("min_frac", 0.3)),
        off_v=float(rule.get("off_v", 0.22)),
        gray_s=float(rule.get("gray_s", 0.20)),
        white_v=float(rule.get("white_v", 0.7)),
    )
    detail.update(verdict.to_dict())
    detail["candidates"] = colors

    color_map = {str(k).lower(): str(v) for k, v in (rule.get("color_map") or {}).items()}
    name = verdict.name
    if name == COLOR_UNCLEAR:
        answer = f"unclear (颜色投票不过阈值: {verdict.detail.get('reason', '')})"
    else:
        answer = color_map.get(name, name)

    raw = f"color({name}, frac={verdict.frac:.2f}) -> {answer}"
    return OpOutcome(answer=answer, raw=raw, detail=detail)


def _op_ocr_text(rgb, item, backend, rule) -> OpOutcome:
    """读字。后端有 `ocr()` 就直接用，否则回落到自由问答。

    保留回落是因为这条 op 应当在 A/B 两个 profile 下都能跑：
    同一份清单能不改就在两条路上跑，才有资格做对比测试。
    """
    if callable(getattr(backend, "ocr", None)):
        text = str(backend.ocr(rgb) or "").strip()
        answer = text or "unclear (OCR 没读出文字)"
        return OpOutcome(
            answer=answer, raw=text, prompt="[op:ocr_text]", detail={"text": text}
        )
    outcome = _run_vqa(rgb, item, backend)
    outcome.detail["fallback"] = "后端没有 ocr()，回落到自由问答"
    return outcome


def _op_vqa(rgb, item, backend, rule) -> OpOutcome:
    """显式走自由问答。用于在以算子为主的清单里标出"这条只能靠语义理解"。"""
    return _run_vqa(rgb, item, backend)


_HANDLERS = {
    "presence": _op_presence,
    "count": _op_count,
    "box_orientation": _op_box_orientation,
    "segment_orientation": _op_segment_orientation,
    "region_color": _op_region_color,
    "ocr_text": _op_ocr_text,
    "vqa": _op_vqa,
}


def describe_ops() -> str:
    """给 CLI 的 --list-ops 用。"""
    lines = ["可用 rule.op:"]
    for name, handler in _HANDLERS.items():
        doc = (handler.__doc__ or "").strip().splitlines()[0]
        lines.append(f"  {name:<20} {doc}")
        lines.append(f"  {'':<20} 参数: {sorted(_OP_KEYS[name] - {'op'})}")
    return "\n".join(lines)


__all__ = [
    "AUTO_COLOR_NAMES",
    "HORIZONTAL",
    "OpOutcome",
    "RuleError",
    "VERTICAL",
    "answer_check",
    "describe_ops",
    "validate_rule",
]
