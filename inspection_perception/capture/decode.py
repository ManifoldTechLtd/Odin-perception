"""图像编解码工具。

三条原则：
1. **后端可插拔且自动降级**：cv2（JetPack 自带，走 libjpeg-turbo）→ PIL → 报错。
   Orin NX 上 cv2 一定存在，开发机上可能只有 PIL，代码两边都能跑。
2. **不依赖 cv_bridge**：cv_bridge 会把 ROS 版本、OpenCV 版本、Python 版本
   三者绑死，是 Jetson 上最常见的环境地狱来源。raw Image 的转换用 numpy
   手写十几行就够，完全可控。
3. **统一输出 RGB**：模型侧（Qwen2.5-VL / PIL）都是 RGB，
   OpenCV 的 BGR 习惯只在本文件内部存在，不外泄。
"""
from __future__ import annotations

import io
from typing import Optional, Tuple

import numpy as np

_CV2 = None
_PIL = None


def _backend() -> str:
    """探测并缓存可用的编解码后端。"""
    global _CV2, _PIL
    if _CV2 is None and _PIL is None:
        try:
            import cv2  # type: ignore

            _CV2 = cv2
            return "cv2"
        except ImportError:
            pass
        try:
            from PIL import Image  # type: ignore

            _PIL = Image
            return "pil"
        except ImportError:
            raise RuntimeError(
                "需要 opencv-python 或 Pillow 之一来做 JPEG 编解码。"
                "Orin NX 上用 JetPack 自带的 python3-opencv，不要 pip 装。"
            )
    return "cv2" if _CV2 is not None else "pil"


def decode_jpeg(buf: bytes) -> np.ndarray:
    """JPEG 字节流 -> HxWx3 uint8 RGB。"""
    if _backend() == "cv2":
        arr = np.frombuffer(buf, dtype=np.uint8)
        bgr = _CV2.imdecode(arr, _CV2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"JPEG 解码失败（{len(buf)} 字节，可能是 PNG 或数据截断）")
        return _CV2.cvtColor(bgr, _CV2.COLOR_BGR2RGB)
    with _PIL.open(io.BytesIO(buf)) as im:
        return np.asarray(im.convert("RGB"))


def encode_jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    """HxWx3 uint8 RGB -> JPEG 字节流。"""
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    if _backend() == "cv2":
        bgr = _CV2.cvtColor(rgb, _CV2.COLOR_RGB2BGR)
        ok, enc = _CV2.imencode(".jpg", bgr, [int(_CV2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise ValueError("JPEG 编码失败")
        return enc.tobytes()
    out = io.BytesIO()
    _PIL.fromarray(rgb).save(out, format="JPEG", quality=quality)
    return out.getvalue()


def raw_to_rgb(data: bytes, height: int, width: int, step: int, encoding: str) -> np.ndarray:
    """sensor_msgs/Image 的裸缓冲区 -> HxWx3 uint8 RGB。

    覆盖工业相机常见编码。Odin 实际用哪种由 `probe_ros_topics.py` 探测确认，
    遇到未覆盖的编码会明确报错而不是静默出错图。

    Args:
        data: msg.data 原始字节。
        height/width: msg.height / msg.width。
        step: msg.step，每行字节数。可能大于 width*channels（行对齐填充）。
        encoding: msg.encoding。
    """
    enc = encoding.lower()
    buf = np.frombuffer(data, dtype=np.uint8)

    if enc in ("rgb8", "bgr8", "rgba8", "bgra8"):
        ch = 4 if enc in ("rgba8", "bgra8") else 3
        img = _reshape_rows(buf, height, width, step, ch)
        if enc.startswith("bgr"):
            img = img[:, :, [2, 1, 0]] if ch == 3 else img[:, :, [2, 1, 0, 3]]
        return np.ascontiguousarray(img[:, :, :3])

    if enc in ("mono8", "8uc1"):
        gray = _reshape_rows(buf, height, width, step, 1)[:, :, 0]
        return np.ascontiguousarray(np.repeat(gray[:, :, None], 3, axis=2))

    if enc in ("mono16", "16uc1"):
        # 深度/高动态灰度：线性归一化到 8bit 仅用于可视化留证，不用于测量。
        u16 = np.frombuffer(data, dtype=np.uint16).reshape(height, -1)[:, :width]
        lo, hi = np.percentile(u16[u16 > 0], [1, 99]) if np.any(u16 > 0) else (0, 1)
        gray = np.clip((u16.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(np.repeat(gray[:, :, None], 3, axis=2))

    if enc in ("yuv422", "yuv422_yuy2", "yuyv", "uyvy"):
        return _yuv422_to_rgb(buf, height, width, step, enc)

    if enc.startswith("bayer"):
        if _backend() != "cv2":
            raise RuntimeError(f"Bayer 编码 {encoding} 的去马赛克需要 opencv")
        code = {
            "bayer_rggb8": _CV2.COLOR_BAYER_BG2RGB,
            "bayer_bggr8": _CV2.COLOR_BAYER_RG2RGB,
            "bayer_gbrg8": _CV2.COLOR_BAYER_GR2RGB,
            "bayer_grbg8": _CV2.COLOR_BAYER_GB2RGB,
        }.get(enc)
        if code is None:
            raise ValueError(f"未支持的 Bayer 编码: {encoding}")
        raw = _reshape_rows(buf, height, width, step, 1)[:, :, 0]
        return _CV2.cvtColor(raw, code)

    raise ValueError(
        f"未支持的 sensor_msgs/Image 编码: {encoding!r}。"
        f"请把 probe_ros_topics.py 的报告发出来，在 raw_to_rgb() 里补上这一支。"
    )


def _reshape_rows(buf: np.ndarray, height: int, width: int, step: int, channels: int) -> np.ndarray:
    """按 step 逐行切片，正确处理行对齐填充。"""
    row_bytes = width * channels
    if step and step != row_bytes:
        return buf[: height * step].reshape(height, step)[:, :row_bytes].reshape(height, width, channels)
    return buf[: height * row_bytes].reshape(height, width, channels)


def _yuv422_to_rgb(buf: np.ndarray, height: int, width: int, step: int, enc: str) -> np.ndarray:
    if _backend() != "cv2":
        raise RuntimeError(f"YUV422 编码 {enc} 的转换需要 opencv")
    packed = _reshape_rows(buf, height, width, step, 2)
    code = _CV2.COLOR_YUV2RGB_UYVY if enc == "uyvy" else _CV2.COLOR_YUV2RGB_YUY2
    return _CV2.cvtColor(packed, code)


def resize_long_side(rgb: np.ndarray, long_side: Optional[int]) -> np.ndarray:
    """按长边等比缩放。

    NX 上的关键省算力手段：Odin 原始分辨率往往远高于 VLM 实际吃进去的分辨率，
    在取图侧就降下来，能同时省掉传输、预处理与视觉塔的开销。
    """
    if not long_side:
        return rgb
    h, w = rgb.shape[:2]
    cur = max(h, w)
    if cur <= long_side:
        return rgb
    scale = long_side / cur
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    if _backend() == "cv2":
        return _CV2.resize(rgb, (nw, nh), interpolation=_CV2.INTER_AREA)
    return np.asarray(_PIL.fromarray(rgb).resize((nw, nh), _PIL.BILINEAR))


def crop_roi(
    rgb: np.ndarray,
    roi: Optional[Tuple[float, float, float, float]],
    pad: float = 0.05,
) -> np.ndarray:
    """按归一化 ROI 裁剪。

    表盘、指示灯、铭牌这类目标在全图里只占几十个像素，而 VLM 会先把图
    缩到 384~768，细节直接被缩没。先裁 ROI 再送进去，是小模型能不能用对
    这些点位的分水岭。

    Args:
        roi: (x0, y0, x1, y1)，均为 0~1。None 时原图返回。
        pad: 向外扩的比例（相对 ROI 尺寸）。留一点上下文比紧贴目标裁更好，
            因为模型需要周边信息才能判断「这是个阀门」而不是「这是个圆形物体」。
    """
    if roi is None:
        return rgb
    x0, y0, x1, y1 = roi
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"ROI 必须是 0~1 且 x0<x1, y0<y1，收到 {roi}")

    if pad:
        dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
        x0, y0 = max(0.0, x0 - dx), max(0.0, y0 - dy)
        x1, y1 = min(1.0, x1 + dx), min(1.0, y1 + dy)

    h, w = rgb.shape[:2]
    px0, py0 = int(round(x0 * w)), int(round(y0 * h))
    px1, py1 = int(round(x1 * w)), int(round(y1 * h))
    # 至少保留 1 像素，避免极窄 ROI 四舍五入后变成空图
    px1, py1 = max(px0 + 1, px1), max(py0 + 1, py1)
    return rgb[py0:py1, px0:px1]
