"""区域主色判定。纯 numpy，不依赖 cv2/PIL。

为什么颜色不交给模型
--------------------
"指示灯是红/绿/黄/灭"这类检查，本质是在问几百个像素的色相，不是在问语义。
让 VLM 回答有三个问题：小模型对颜色词经常张冠李戴、同一张图两次问可能不同答案、
判错了没法解释。而色相统计是**确定性**的：同一张图永远同一个结论，判错了能直接
把票数和阈值打出来看是哪一步不对。

Florence-2 尤其需要这一层：它不能自由问答，原先的实现是「逐个颜色词跑一次开词表
检测」——4 个选项要 4 次推理，而且检测器对"green"这种形容词本来就不敏感。
换成先定位一次、再算色相，推理次数从 N 降到 1，准确率反而更高。

三个必须处理的现场问题
----------------------
**① 指示灯只占 ROI 的一小块。** 直接对整个 ROI 求平均色，会被面板底色（通常是
灰色或米色）稀释，永远算出"灰"。所以先按亮度取最亮的那一撮像素再投票——
亮着的灯一定是画面里最亮的东西。

**② "灭"必须能表达。** 灯灭时最亮的像素就是灯罩本身，色相是随机的。
所以先用绝对亮度阈值判"灭"，通过了才谈色相。

**③ 红色跨越色相环 0°。** 红是 [345,360) ∪ [0,15)，写成单区间必然漏一半。
本模块的色带一律是区间**列表**。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geometry import Box

UNCLEAR = "unclear"
OFF = "off"
WHITE = "white"
GRAY = "gray"

HueRange = Tuple[float, float]


@dataclass
class ColorBand:
    """一个有彩色色带。

    `min_s` / `min_v` 是该色带自己的门槛：深红（暗红漆面）和亮红（点亮的红灯）
    饱和度差很多，允许逐色带调而不是全局一个阈值。
    """

    name: str
    hues: List[HueRange]
    min_s: float = 0.35
    min_v: float = 0.25

    def matches(self, h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
        ok = (s >= self.min_s) & (v >= self.min_v)
        hue_ok = np.zeros_like(ok)
        for lo, hi in self.hues:
            hue_ok |= (h >= lo) & (h < hi)
        return ok & hue_ok


# 默认色带。范围按 HSV 色相角（度）给，边界刻意留窄一点：
# 宁可判 unclear 让人看一眼，也不要把橙色灯读成红灯——这两者在配电柜上语义不同。
DEFAULT_PALETTE: List[ColorBand] = [
    ColorBand("red", [(0.0, 12.0), (348.0, 360.0)]),
    ColorBand("orange", [(12.0, 40.0)]),
    ColorBand("yellow", [(40.0, 70.0)]),
    ColorBand("green", [(75.0, 165.0)]),
    ColorBand("cyan", [(165.0, 195.0)]),
    ColorBand("blue", [(200.0, 255.0)]),
    ColorBand("purple", [(260.0, 345.0)]),
]

_PALETTE_BY_NAME = {band.name: band for band in DEFAULT_PALETTE}


@dataclass
class ColorVerdict:
    """判定结果。`detail` 是给人排查用的，会原样写进存证。"""

    name: str
    """胜出的颜色名，或 "off" / "white" / "gray" / "unclear"。"""

    frac: float = 0.0
    """胜出颜色在参与投票的像素中的占比。"""

    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"color": self.name, "frac": round(self.frac, 3), **self.detail}


def rgb_to_hsv(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """uint8 RGB -> (h[0,360), s[0,1], v[0,1])，逐像素向量化。

    自己算而不用 `colorsys`（只能标量）或 cv2（色相被压到 0~179 且要求 BGR）：
    十几行就够，还免掉一个在 venv 里不一定存在的依赖。
    """
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    v = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    chroma = v - mn

    s = np.where(v > 0, chroma / np.maximum(v, 1e-6), 0.0)

    h = np.zeros_like(v)
    nz = chroma > 1e-6
    # 分段公式，只在 chroma>0 的像素上算，灰像素色相无意义留 0
    rmax = nz & (v == r)
    gmax = nz & (v == g) & ~rmax
    bmax = nz & ~rmax & ~gmax
    c = np.maximum(chroma, 1e-6)
    h[rmax] = ((g[rmax] - b[rmax]) / c[rmax]) % 6.0
    h[gmax] = (b[gmax] - r[gmax]) / c[gmax] + 2.0
    h[bmax] = (r[bmax] - g[bmax]) / c[bmax] + 4.0
    h = (h * 60.0) % 360.0
    return h, s, v


def resolve_palette(names: Optional[Sequence[str]] = None) -> List[ColorBand]:
    """按名字挑出要参与投票的色带。

    只投清单里真正可能出现的颜色，能显著减少误判：一个只有红/绿两态的指示灯，
    把橙、黄、青一起放进来投票，只会让边缘色相把票分走导致 unclear。
    """
    if not names:
        return list(DEFAULT_PALETTE)
    bands: List[ColorBand] = []
    unknown: List[str] = []
    for name in names:
        band = _PALETTE_BY_NAME.get(str(name).strip().lower())
        if band is None:
            unknown.append(str(name))
        else:
            bands.append(band)
    if unknown:
        raise ValueError(
            f"未知颜色名 {unknown}，可选 {sorted(_PALETTE_BY_NAME)}"
            f"（off/white/gray 是自动判定的，不用写进来）"
        )
    return bands


def classify_region(
    rgb: np.ndarray,
    box: Optional[Box] = None,
    colors: Optional[Sequence[str]] = None,
    bright_percentile: float = 75.0,
    min_frac: float = 0.3,
    off_v: float = 0.22,
    gray_s: float = 0.20,
    white_v: float = 0.7,
) -> ColorVerdict:
    """判定区域主色。

    Args:
        rgb: HxWx3 uint8。
        box: 只看这个框内的像素。None 时看整张（调用方通常已经裁过 ROI）。
        colors: 参与投票的颜色名；None 时用全部默认色带。
        bright_percentile: 只用亮度排在这个百分位以上的像素投票。
            指示灯只占 ROI 一小块，不做这一步会被面板底色稀释。
            设 0 表示用全部像素（适合判断大面积漆面颜色）。
        min_frac: 胜出颜色至少要占这么多票，否则判 unclear 而不是硬选一个。
        off_v: 参与投票的像素亮度中位数低于此值 -> "off"（灯灭/无光）。
        gray_s: 饱和度低于此值算无彩色。
        white_v: 无彩色且亮度高于此值 -> "white"，否则 "gray"。

    Returns:
        `ColorVerdict`。`detail` 里带各色票数与亮度统计，便于回溯阈值是否合适。
    """
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        return ColorVerdict(UNCLEAR, detail={"error": "需要 HxWx3 RGB"})

    region = rgb
    if box is not None:
        x0, y0, x1, y1 = box.to_int_slice(rgb.shape[1], rgb.shape[0])
        region = rgb[y0:y1, x0:x1]
    if region.size == 0:
        return ColorVerdict(UNCLEAR, detail={"error": "区域为空"})

    h, s, v = rgb_to_hsv(region)
    h, s, v = h.ravel(), s.ravel(), v.ravel()

    if bright_percentile and bright_percentile > 0:
        thr = float(np.percentile(v, bright_percentile))
        mask = v >= thr
        if not mask.any():  # 全图同一亮度时 percentile 等于最大值，兜底用全部像素
            mask = np.ones_like(v, dtype=bool)
        h, s, v = h[mask], s[mask], v[mask]

    total = int(v.size)
    stats = {
        "pixels": total,
        "v_median": round(float(np.median(v)), 3),
        "s_median": round(float(np.median(s)), 3),
    }

    # 先判"灭"：灯灭时最亮的像素就是灯罩，色相是噪声，不能拿去投票
    if stats["v_median"] < off_v:
        return ColorVerdict(OFF, 1.0, {**stats, "reason": f"亮度中位数 < {off_v}"})

    bands = resolve_palette(colors)
    votes: Dict[str, int] = {}
    for band in bands:
        votes[band.name] = int(band.matches(h, s, v).sum())
    achromatic = int((s < gray_s).sum())
    stats["votes"] = votes
    stats["achromatic"] = achromatic

    best_name, best_count = "", 0
    for name, count in votes.items():
        if count > best_count:
            best_name, best_count = name, count

    best_frac = best_count / total if total else 0.0
    achromatic_frac = achromatic / total if total else 0.0

    # 无彩色压倒性多数时不要硬凑一个颜色出来
    if achromatic_frac > best_frac and achromatic_frac >= min_frac:
        name = WHITE if stats["v_median"] >= white_v else GRAY
        return ColorVerdict(name, achromatic_frac, {**stats, "reason": "无彩色占多数"})

    if best_frac < min_frac:
        return ColorVerdict(
            UNCLEAR,
            best_frac,
            {**stats, "reason": f"最高票 {best_name or '无'} 仅占 {best_frac:.2f} < {min_frac}"},
        )
    return ColorVerdict(best_name, best_frac, stats)


__all__ = [
    "DEFAULT_PALETTE",
    "GRAY",
    "OFF",
    "UNCLEAR",
    "WHITE",
    "ColorBand",
    "ColorVerdict",
    "classify_region",
    "resolve_palette",
    "rgb_to_hsv",
]
