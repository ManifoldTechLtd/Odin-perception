"""从本地图片文件取图。

用途有两个，都很实际：
    1. **离线回归**：把现场采集的图存成一批，反复调 prompt 与判定阈值，
       不用每次都开机器人。这是调准确率唯一可行的方式。
    2. **联调解耦**：ROS 那边还没通、模型这边想先跑，用它顶上。

只依赖标准库 + `decode`，没有 ROS 也没有网络。
"""
from __future__ import annotations

import os
from typing import List, Optional

from .base import Frame

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class FileFrameSource:
    """把本地图片（单个文件或目录）当作取图源。

    Args:
        path: 图片文件或包含图片的目录。目录会按文件名排序遍历。
        loop: 取完最后一张后是否从头再来。False 时之后返回 None，
            这样 `while src.get_latest()` 能自然终止，适合批量回归。
    """

    def __init__(self, path: str, loop: bool = True) -> None:
        self.paths: List[str] = self._collect(path)
        if not self.paths:
            raise FileNotFoundError(f"{path} 下没有找到图片（支持 {', '.join(_IMAGE_EXTS)}）")
        self.loop = loop
        self._index = 0
        self._seq = 0

    @staticmethod
    def _collect(path: str) -> List[str]:
        if os.path.isfile(path):
            return [path]
        if os.path.isdir(path):
            return sorted(
                os.path.join(path, name)
                for name in os.listdir(path)
                if name.lower().endswith(_IMAGE_EXTS)
            )
        raise FileNotFoundError(f"路径不存在: {path}")

    def get_latest(self, timeout_s: float = 2.0, max_age_ms: float = 1000.0) -> Optional[Frame]:
        """返回下一张图。

        文件源没有「过期」概念，`max_age_ms` 被忽略——它读出来的帧
        `recv_ns` 就是当下，永远新鲜。
        """
        if self._index >= len(self.paths):
            if not self.loop:
                return None
            self._index = 0

        path = self.paths[self._index]
        self._index += 1
        self._seq += 1

        with open(path, "rb") as fh:
            data = fh.read()

        # PNG/BMP 也塞进 jpeg 字段：Frame.as_rgb() 走的是通用解码，
        # 字段名叫 jpeg 只是历史习惯，实际存的是「已编码的图像字节」。
        return Frame(
            seq=self._seq,
            frame_id="file",
            topic=path,
            encoding=os.path.splitext(path)[1].lstrip(".").lower(),
            jpeg=data,
        )

    def wait_for_first_frame(self, timeout_s: float = 10.0) -> bool:
        return True

    def stats(self) -> dict:
        return {"frames_total": self._seq, "files": len(self.paths), "loop": self.loop}

    def close(self) -> None:
        pass
