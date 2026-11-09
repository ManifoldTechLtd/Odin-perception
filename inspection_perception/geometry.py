"""检测框与多边形的几何算子。纯 numpy，无模型、无 ROS、无 cv2。

为什么需要这一层
----------------
Florence-2 不能自由问答，但它**能框、能分割**。巡检里大量"是非/状态"判断，
本质上不是语言问题而是几何问题：

    阀门是否关闭   ->  手柄长轴是否接近竖直
    断路器开还是合  ->  拨杆长轴角度落在哪个区间
    仪表指针指向   ->  指针长轴角度

把这些交给模型"用语言回答"，等于让它做它最不擅长的事——即便是 Moondream2
这类能问答的模型，对"垂直/水平"这种空间关系也经常答反。而先让模型只做它擅长的
定位（给框、给轮廓），再用确定性算子算角度，结果是**可复现、可单测、可解释的**：
判错了能指出是框错了还是角度阈值不对，而不是笼统地说"模型效果不好"。

这也是本文件能在没有现场图的情况下就完成验证的原因：角度计算是确定性的，
用合成图（画一根已知角度的杆）可以完全证明正确性，只剩"框得准不准"需要真实图像。

放在包根目录而不是 `checks/` 或 `vlm/` 下
------------------------------------------
`Box` 是 `vlm/`（产出）和 `checks/`（消费）之间的共享结构。放进任何一侧都会
制造反向依赖。它本身只是几何原语，不含任何业务语义，所以放在包根做叶子模块。

角度约定
--------
一律用**视觉直觉**的方向：0° = 水平（指向右），90° = 竖直（指向上），
逆时针为正，取值归约到 [0, 180)。图像坐标系 y 向下增长，本文件内部已经翻转过，
调用方不需要关心。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

VERTICAL = "vertical"
HORIZONTAL = "horizontal"
AMBIGUOUS = "ambiguous"

Point = Tuple[float, float]


@dataclass
class Box:
    """一个轴对齐检测框，坐标是**像素**（模型 post_process 出来就是像素）。

    `score` 是 Optional 而不是默认 1.0：Florence-2 的检测输出**根本不带置信度**。
    给个假的 1.0 会诱使上层写出 `min_score` 之类看着严谨、实际永远不生效的过滤条件。
    留 None 才能让「这个后端没有置信度」这件事在类型上显形。
    """

    x0: float
    y0: float
    x1: float
    y1: float
    label: str = ""
    score: Optional[float] = None

    def __post_init__(self) -> None:
        # 有些后端会把坐标顺序反过来给，统一规整，避免下游算出负宽高
        if self.x1 < self.x0:
            self.x0, self.x1 = self.x1, self.x0
        if self.y1 < self.y0:
            self.y0, self.y1 = self.y1, self.y0

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def area_frac(self, image_w: float, image_h: float) -> float:
        """占整图面积的比例。用于过滤「整图大小」的幻觉框。"""
        total = float(image_w) * float(image_h)
        return self.area / total if total > 0 else 0.0

    def clip(self, image_w: float, image_h: float) -> "Box":
        return Box(
            max(0.0, min(self.x0, image_w)),
            max(0.0, min(self.y0, image_h)),
            max(0.0, min(self.x1, image_w)),
            max(0.0, min(self.y1, image_h)),
            self.label,
            self.score,
        )

    def to_int_slice(self, image_w: int, image_h: int) -> Tuple[int, int, int, int]:
        """转成可直接切片的整数 (x0, y0, x1, y1)，至少 1 像素宽高。"""
        b = self.clip(image_w, image_h)
        x0, y0 = int(math.floor(b.x0)), int(math.floor(b.y0))
        x1, y1 = int(math.ceil(b.x1)), int(math.ceil(b.y1))
        return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "x0": round(self.x0, 1),
            "y0": round(self.y0, 1),
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "label": self.label,
        }
        if self.score is not None:
            data["score"] = round(float(self.score), 3)
        return data


def iou(a: Box, b: Box) -> float:
    """交并比。"""
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def dedupe(boxes: Sequence[Box], iou_thr: float = 0.6) -> List[Box]:
    """合并重复框，按面积从大到小保留。

    **计数类检查必须先过这一步。** Florence-2 的检测是生成式的，同一个目标
    经常吐出两三个高度重叠的框（尤其是让它找复数名词时）。不去重的话
    「数几个灭火器」会稳定地多数出来一两个，而这种错在现场极难察觉——
    数字看着合理，只是偏大。
    """
    kept: List[Box] = []
    for box in sorted(boxes, key=lambda b: b.area, reverse=True):
        if all(iou(box, other) < iou_thr for other in kept):
            kept.append(box)
    return kept


def filter_boxes(
    boxes: Sequence[Box],
    image_w: float,
    image_h: float,
    min_area_frac: float = 0.0,
    max_area_frac: float = 1.0,
    label_contains: str = "",
) -> List[Box]:
    """按面积占比与标签过滤。

    `max_area_frac` 是对付 Florence-2 幻觉的主力手段，不是可选的精致优化：
    它找不到目标时最典型的行为是**返回一个几乎覆盖整图的框**。对
    「期望没有渗漏」这类 expect=no 的检查项，不卡上限就会稳定误报 FAIL。
    默认留 1.0（不过滤）是为了不悄悄改变语义，但清单里应当显式写 0.9 左右。
    """
    out: List[Box] = []
    for box in boxes:
        frac = box.area_frac(image_w, image_h)
        if frac < min_area_frac or frac > max_area_frac:
            continue
        if label_contains and label_contains.lower() not in box.label.lower():
            continue
        out.append(box)
    return out


def largest(boxes: Sequence[Box]) -> Optional[Box]:
    """取面积最大的框。姿态判断只该看一个目标，多框时以最大的为准。"""
    return max(boxes, key=lambda b: b.area) if boxes else None


# --------------------------------------------------------------------- 姿态
def orientation_from_box(box: Box, ratio: float = 1.3) -> str:
    """仅凭外接框的长宽比判竖直/水平。

    Returns:
        "vertical" | "horizontal" | "ambiguous"

    `ambiguous` 是必须存在的第三态，不能强行二分：轴对齐外接框对**斜着**的
    杆状物完全无能为力——45° 的手柄外接框是正方形，长宽比 1.0。此时返回
    "ambiguous"，上层判 UNKNOWN 交人复核，比瞎猜一个方向安全得多。
    要处理斜置目标就用 `polygon_orientation()`（需要后端支持分割）。
    """
    if box.w <= 0 or box.h <= 0:
        return AMBIGUOUS
    if box.h / box.w >= ratio:
        return VERTICAL
    if box.w / box.h >= ratio:
        return HORIZONTAL
    return AMBIGUOUS


def polygon_area(points: Sequence[Point]) -> float:
    """多边形面积（鞋带公式，取绝对值，不关心顶点旋向）。

    用于在多个分割轮廓里选主目标：分割常会带出碎片（反光、遮挡切开的一小块），
    拿碎片去算主轴会得到完全无关的角度。
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_bbox(points: Sequence[Point], label: str = "") -> Box:
    """轮廓的外接框。存证里要把轮廓位置记下来，但存几百个顶点没意义。"""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return Box(0.0, 0.0, 0.0, 0.0, label)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return Box(float(x0), float(y0), float(x1), float(y1), label)


def rasterize_polygon(points: Sequence[Point], grid: int = 128) -> np.ndarray:
    """把多边形栅格化成布尔掩码（奇偶规则）。

    栅格化到固定分辨率再算矩，而不是直接对顶点做 PCA：顶点是不均匀采样的
    （长边可能只有两个顶点，拐角处挤了十几个），直接对顶点求协方差会被顶点
    密度带偏，算出来的主轴可能偏几十度。栅格化等价于对**面积**均匀采样。

    128 的网格在 Orin NX 上耗时是微秒级，相对一次模型推理可以忽略。
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        raise ValueError(f"多边形至少要 3 个 (x, y) 顶点，收到 shape={pts.shape}")

    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    span_x, span_y = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    # 保持长宽比：否则栅格化本身就会把形状拉成方的，主轴角度直接失真
    scale = grid / max(span_x, span_y)
    nx = max(2, int(round(span_x * scale)))
    ny = max(2, int(round(span_y * scale)))

    # 取像素中心，避免边界上的顶点落在网格线上导致奇偶判定抖动
    xs = x0 + (np.arange(nx) + 0.5) * span_x / nx
    ys = y0 + (np.arange(ny) + 0.5) * span_y / ny
    gx, gy = np.meshgrid(xs, ys)

    inside = np.zeros((ny, nx), dtype=bool)
    n = pts.shape[0]
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        if ay == by:  # 水平边不参与水平射线的奇偶计数
            continue
        straddles = (ay > gy) != (by > gy)
        x_cross = ax + (gy - ay) * (bx - ax) / (by - ay)
        inside ^= straddles & (gx < x_cross)

    return inside


def mask_principal_axis(mask: np.ndarray) -> Tuple[float, float]:
    """求布尔掩码的主轴方向与细长度。

    Returns:
        (angle_deg, elongation)
        angle_deg  : 主轴方向，[0, 180)，0=水平 90=竖直，逆时针为正（视觉方向）。
        elongation : 长轴/短轴，圆形约 1.0，越大越细长。**必须一起用**——
                     圆形目标的主轴方向是数值噪声，只看角度会得到一个看似
                     确定的答案，用 elongation 卡掉才不会拿噪声当结论。
    """
    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        return 0.0, 1.0

    x = xs.astype(np.float64) - xs.mean()
    # 图像 y 向下，翻正号让角度符合视觉直觉（逆时针为正）
    y = -(ys.astype(np.float64) - ys.mean())

    cov = np.cov(np.vstack([x, y]))
    if not np.all(np.isfinite(cov)):
        return 0.0, 1.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    major = eigvecs[:, order[0]]

    angle = math.degrees(math.atan2(major[1], major[0])) % 180.0
    lo = max(float(eigvals[1]), 1e-12)
    elongation = math.sqrt(max(float(eigvals[0]), 0.0) / lo)
    return angle, elongation


def polygon_orientation(points: Sequence[Point], grid: int = 128) -> Tuple[float, float]:
    """分割轮廓 -> (主轴角度, 细长度)。见 `mask_principal_axis` 的约定。"""
    return mask_principal_axis(rasterize_polygon(points, grid=grid))


def angle_to_orientation(
    angle_deg: float,
    elongation: float,
    tol_deg: float = 30.0,
    min_elongation: float = 1.25,
) -> str:
    """把角度归到竖直/水平，落在中间地带或目标不够细长时返回 ambiguous。

    `tol_deg` 默认 30° 而不是 45°：留一条死区带。若按 45° 一刀切，44° 的手柄
    会被判成"水平"、46° 判成"竖直"，同一个没动过的阀门在两次巡检里给出相反结论，
    比给 UNKNOWN 让人看一眼糟糕得多。
    """
    if elongation < min_elongation:
        return AMBIGUOUS
    a = angle_deg % 180.0
    if min(a, 180.0 - a) <= tol_deg:
        return HORIZONTAL
    if abs(a - 90.0) <= tol_deg:
        return VERTICAL
    return AMBIGUOUS


def angle_delta(a: float, b: float) -> float:
    """两个无向角之间的差，取值 [0, 90]。用于「指针是否指在某个方向」。"""
    d = abs((a % 180.0) - (b % 180.0)) % 180.0
    return min(d, 180.0 - d)


def relative_position(box: Box, reference: Box) -> str:
    """`box` 相对 `reference` 在哪边。按中心偏移量大的那个轴定，图像 y 向下。

    用于"液位是否高于刻度线""手柄是否在阀体上方"这类检查。
    """
    dx = box.cx - reference.cx
    dy = box.cy - reference.cy
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "below" if dy > 0 else "above"


__all__ = [
    "AMBIGUOUS",
    "Box",
    "HORIZONTAL",
    "VERTICAL",
    "angle_delta",
    "angle_to_orientation",
    "dedupe",
    "filter_boxes",
    "iou",
    "largest",
    "mask_principal_axis",
    "orientation_from_box",
    "polygon_area",
    "polygon_bbox",
    "polygon_orientation",
    "rasterize_polygon",
    "relative_position",
]
