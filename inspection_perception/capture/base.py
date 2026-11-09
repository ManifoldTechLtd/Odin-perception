"""取图层的统一契约。

设计要点（针对 Orin NX）：
1. `Frame` 同时持有 `jpeg` 与 `rgb` 两种表示，且都允许为 None——
   谁需要谁解码，避免在 NX 上做无人使用的 JPEG 解码。
2. `FrameSource` 只有「拿最新一帧」这一个语义，没有队列。
   巡检检查是「到点位后取当前画面」，历史帧没有价值，
   积压队列在 NX 上只会拖高延迟和内存。
3. 实现方可以是 ROS 节点（同进程）或 HTTP 客户端（跨进程/跨机器），
   上层 pipeline 不感知差异。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, runtime_checkable

import numpy as np


@dataclass
class Frame:
    """一帧图像及其溯源信息。

    Attributes:
        seq: 本进程内自增序号，用于判断是否拿到了新帧。
        stamp_ns: ROS 消息头时间戳（纳秒）。0 表示上游未填。
        recv_ns: 本机接收时刻（`time.time_ns()`），用于算端到端时延。
        frame_id: ROS header.frame_id，证据留档与外参对齐时需要。
        topic: 实际来源话题，便于排查「拿到的是 raw 还是 compressed」。
        width/height: 图像尺寸。raw 话题直接给出；compressed 话题在解码前为 0。
        encoding: 原始编码字符串（如 bgr8 / rgb8 / jpeg）。
        jpeg: JPEG 字节流。来自 compressed 话题时是原始字节，零成本。
        rgb: HxWx3 uint8 RGB 数组。VLM 侧统一用 RGB，不用 BGR。
    """

    seq: int
    stamp_ns: int = 0
    recv_ns: int = field(default_factory=time.time_ns)
    frame_id: str = ""
    topic: str = ""
    width: int = 0
    height: int = 0
    encoding: str = ""
    jpeg: Optional[bytes] = None
    rgb: Optional[np.ndarray] = None

    @property
    def age_ms(self) -> float:
        """距离接收时刻已过去多少毫秒。用于拒绝陈旧帧。"""
        return (time.time_ns() - self.recv_ns) / 1e6

    def as_rgb(self) -> np.ndarray:
        """按需拿到 RGB 数组；只有此时才付出 JPEG 解码代价。"""
        if self.rgb is None:
            if self.jpeg is None:
                raise ValueError(f"Frame(seq={self.seq}) 既无 rgb 也无 jpeg")
            from .decode import decode_jpeg

            self.rgb = decode_jpeg(self.jpeg)
            self.height, self.width = self.rgb.shape[:2]
        return self.rgb

    def as_jpeg(self, quality: int = 90) -> bytes:
        """按需拿到 JPEG 字节流（证据留档、跨进程传输用）。"""
        if self.jpeg is None:
            if self.rgb is None:
                raise ValueError(f"Frame(seq={self.seq}) 既无 rgb 也无 jpeg")
            from .decode import encode_jpeg

            self.jpeg = encode_jpeg(self.rgb, quality=quality)
        return self.jpeg

    def describe(self) -> str:
        kind = []
        if self.jpeg is not None:
            kind.append(f"jpeg({len(self.jpeg)}B)")
        if self.rgb is not None:
            kind.append(f"rgb{self.rgb.shape}")
        return (
            f"Frame#{self.seq} {self.width}x{self.height} {self.encoding} "
            f"from={self.topic} frame_id={self.frame_id} "
            f"age={self.age_ms:.0f}ms [{'+'.join(kind) or 'empty'}]"
        )


@runtime_checkable
class FrameSource(Protocol):
    """取图源统一接口。"""

    def get_latest(self, timeout_s: float = 2.0, max_age_ms: float = 1000.0) -> Optional[Frame]:
        """取当前最新一帧。

        Args:
            timeout_s: 没有可用帧时最多等待多久。
            max_age_ms: 帧的最大可接受年龄；超过则继续等待更新的帧。
                设为 0 表示不做新鲜度检查。

        Returns:
            Frame，或超时返回 None。
        """
        ...

    def close(self) -> None:
        """释放资源。"""
        ...


class LatestFrameStore:
    """「只保留最新一帧」的线程安全存储。

    ROS 1（rospy）与 ROS 2（rclpy）的订阅回调都往这里写，取图侧从这里读。
    把这段逻辑抽出来，两个 ROS 版本的适配器就只剩「消息 → Frame」的转换差异。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[Frame] = None
        self._seq = 0
        # NX 上是 Python 3.8，虽然有 `from __future__ import annotations` 兜着，
        # 但 typing.get_type_hints 会在求值时炸，统一用 typing.Dict 写。
        self._counts: Dict[str, int] = {}
        self._first_recv_ns: Optional[int] = None

    def put(self, frame: Frame, kind: str) -> None:
        with self._lock:
            self._seq += 1
            frame.seq = self._seq
            self._latest = frame
            self._counts[kind] = self._counts.get(kind, 0) + 1
            if self._first_recv_ns is None:
                self._first_recv_ns = frame.recv_ns

    def get_latest(self, timeout_s: float = 2.0, max_age_ms: float = 1000.0) -> Optional[Frame]:
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                frame = self._latest
            if frame is not None and (max_age_ms <= 0 or frame.age_ms <= max_age_ms):
                return frame
            if time.monotonic() >= deadline:
                # 不做新鲜度检查时，超时也把手里这帧交出去（可能为 None）
                return frame if max_age_ms <= 0 else None
            time.sleep(0.01)

    def wait_for_first(self, timeout_s: float = 10.0) -> bool:
        """等到第一帧到达。启动阶段用来给出明确的「收不到图」诊断。"""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest is not None:
                    return True
            time.sleep(0.05)
        return False

    def stats(self) -> dict:
        with self._lock:
            elapsed = (time.time_ns() - self._first_recv_ns) / 1e9 if self._first_recv_ns else 0.0
            return {
                "frames_total": self._seq,
                "frames_by_kind": dict(self._counts),
                "elapsed_s": round(elapsed, 2),
                "fps": round(self._seq / elapsed, 2) if elapsed > 0.5 else None,
            }
