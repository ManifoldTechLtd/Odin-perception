"""把检查结果画到图上：检测框 + 底部文字结论。

**这个模块解决什么**

`frame_server` 早就提供了 `POST /annotate` 入口、`ros1_odin_node` 也实现了
`annotate()` 发布到 `/inspection/annotated/compressed`，但 pipeline 从来没调用过它们，
所以话题一直是空的——"能看见框和文字"这件事只存在于预期里。

本模块补上最后一环：把每条检查的**框**和**文字结论**画到模型实际看到的那张图上，
再交给调用方 POST 回去。

**排版约定（现场可调）**

- 文字面板固定在图片**底部**，半透明黑底 + 白字，不遮挡检测目标；
- 最多 `max_lines` 行（默认 3），超出时最后一行用 `…` 收尾，不无限变高；
- 字号按图片长边自适应，再用 `scale` 乘一个系数（现场嫌小就调 scale，不用改代码）；
- 字体优先 Noto/WQY 中文字体，都没有时回落 Pillow 默认字体（此时中文会显示成方块，
  但英文与数字仍可读）。
"""
from __future__ import annotations

import io
import os
import textwrap
from typing import Any, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

# 按优先级探测字体：NX/Ubuntu 上的 Noto CJK 最完整，Windows 与 macOS 的路径顺带兜住
_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)

# 框颜色：按状态区分，现场一眼能看出合格与否
_COLOR_PASS = (0, 200, 0)
_COLOR_FAIL = (230, 40, 40)
_COLOR_UNKNOWN = (235, 170, 0)
_COLOR_ERROR = (150, 150, 150)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:  # 字体损坏/不支持的格式，继续找下一个
                continue
    return ImageFont.load_default()


def _status_color(status: Any) -> Tuple[int, int, int]:
    name = getattr(status, "value", status)
    name = str(name)
    if "PASS" in name:
        return _COLOR_PASS
    if "FAIL" in name:
        return _COLOR_FAIL
    if "ERROR" in name:
        return _COLOR_ERROR
    return _COLOR_UNKNOWN


def _iter_boxes(detail: Any) -> List[Sequence[float]]:
    """从算子返回的 detail 里尽量拗出框，格式统一成 [x0, y0, x1, y1]。

    不同算子给的键名不一致（box / boxes / bbox / detected），这里都认；
    拗不出来就返回空列表——画不出框不该让整条检查失败。
    """
    out: List[Sequence[float]] = []
    if not isinstance(detail, dict):
        return out

    def _push(v: Any) -> None:
        if isinstance(v, (list, tuple)) and len(v) == 4 and all(
            isinstance(x, (int, float)) for x in v
        ):
            out.append(v)

    for key in ("box", "bbox", "person_box", "target_box"):
        if key in detail:
            _push(detail[key])
    for key in ("boxes", "detected", "kept"):
        seq = detail.get(key)
        if isinstance(seq, (list, tuple)):
            for item in seq:
                if isinstance(item, dict):
                    for k in ("box", "bbox"):
                        if k in item:
                            _push(item[k])
                else:
                    _push(item)
    return out


# 公开别名：pipeline 等外部模块需要从 detail 里取框
def iter_boxes(detail: Any) -> List[Sequence[float]]:
    return _iter_boxes(detail)


def draw_boxes(
    draw: ImageDraw.ImageDraw,
    boxes: Sequence[Sequence[float]],
    color: Tuple[int, int, int],
    width: int,
) -> None:
    for b in boxes:
        try:
            x0, y0, x1, y1 = (float(v) for v in b)
        except (TypeError, ValueError):
            continue
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)


def draw_text_panel(
    img: Image.Image,
    lines: Sequence[str],
    max_lines: int = 3,
    scale: float = 1.0,
) -> Image.Image:
    """在图片底部画文字面板。返回新图（不修改入参）。"""
    lines = [ln for ln in (s.strip() for s in lines) if ln]
    if not lines:
        return img

    base = img.convert("RGB")
    w, h = base.size
    font_size = max(10, int(max(w, h) * 0.022 * max(0.3, scale)))
    font = _load_font(font_size)

    # 先按宽度折行，再截断到 max_lines（超出用 … 收尾）
    char_per_line = max(8, int(w / (font_size * 0.62)))
    wrapped: List[str] = []
    for ln in lines:
        wrapped.extend(textwrap.wrap(ln, width=char_per_line) or [ln])
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        wrapped[-1] = wrapped[-1][: max(1, char_per_line - 1)] + "…"

    pad = max(4, font_size // 3)
    line_h = int(font_size * 1.28)
    panel_h = len(wrapped) * line_h + pad * 2

    canvas = Image.new("RGB", (w, h + panel_h), (0, 0, 0))
    canvas.paste(base, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, h, w, h + panel_h], fill=(0, 0, 0))

    y = h + pad
    for ln in wrapped:
        draw.text((pad, y), ln, font=font, fill=(255, 255, 255))
        y += line_h
    return canvas


def render_result(
    rgb: Any,
    lines: Sequence[str],
    boxes: Optional[Sequence[Sequence[float]]] = None,
    status: Any = None,
    max_lines: int = 3,
    scale: float = 1.0,
    box_width: int = 3,
) -> Image.Image:
    """画框 + 底部文字，返回合成后的图。"""
    if hasattr(rgb, "shape"):  # numpy 数组（BGR/RGB 由调用方保证）
        img = Image.fromarray(rgb)
    else:
        img = rgb
    img = img.convert("RGB")
    if boxes:
        draw = ImageDraw.Draw(img)
        draw_boxes(draw, boxes, _status_color(status), box_width)
    return draw_text_panel(img, lines, max_lines=max_lines, scale=scale)


def encode_jpeg(img: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
