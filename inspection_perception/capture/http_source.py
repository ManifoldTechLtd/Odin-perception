"""HTTP 取图客户端：推理进程侧的 `FrameSource` 实现。

只用标准库 urllib，因此可以放进任何 Python 环境（包括 VLM-FO1 的 3.10 venv），
不需要 rclpy、不需要 requests。
"""
from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from .base import Frame


class HttpFrameSource:
    """从 `frame_server` 拉取最新帧（结构化实现 `FrameSource`）。

    Example:
        >>> src = HttpFrameSource("http://127.0.0.1:8730", long_side=1280)
        >>> frame = src.get_latest()
        >>> rgb = frame.as_rgb()
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8730",
        long_side: Optional[int] = None,
        quality: int = 90,
        request_timeout_s: float = 5.0,
    ) -> None:
        """
        Args:
            base_url: frame_server 地址。NX 上跑取图、开发机跑推理时填 NX 的 IP。
            long_side: 请求服务端按长边缩放后再传。跨机时能显著降低传输量，
                但会让服务端付出一次解码+编码；同机时建议留空走原始字节透传。
        """
        self._base = base_url.rstrip("/")
        self._long_side = long_side
        self._quality = quality
        self._timeout = request_timeout_s

    def get_latest(self, timeout_s: float = 2.0, max_age_ms: float = 1000.0) -> Optional[Frame]:
        params = {"timeout_s": timeout_s, "max_age_ms": max_age_ms, "quality": self._quality}
        if self._long_side:
            params["long_side"] = self._long_side
        url = f"{self._base}/latest.jpg?{urllib.parse.urlencode(params)}"

        try:
            with urllib.request.urlopen(url, timeout=max(self._timeout, timeout_s + 2)) as resp:
                body = resp.read()
                h = resp.headers
        except urllib.error.HTTPError as exc:
            if exc.code == 503:  # 服务端在线但暂时没有新鲜帧
                return None
            raise
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(
                f"取图服务不可达: {self._base} ({exc})。"
                f"确认 ros_odin_node --serve 是否在运行，跨机时确认端口是否放通。"
            ) from exc

        def _int(name: str) -> int:
            try:
                return int(h.get(name, "0"))
            except ValueError:
                return 0

        age_ms = float(h.get("X-Frame-Age-Ms", "0") or 0)
        return Frame(
            seq=_int("X-Frame-Seq"),
            stamp_ns=_int("X-Frame-Stamp-Ns"),
            # 用服务端上报的 age 反推接收时刻，让跨进程后的 age_ms 依然近似可信
            recv_ns=time.time_ns() - int(age_ms * 1e6),
            frame_id=h.get("X-Frame-Id", ""),
            topic=h.get("X-Frame-Topic", ""),
            encoding=h.get("X-Frame-Encoding", "jpeg"),
            jpeg=body,
        )

    def meta(self) -> dict:
        """拉取服务端状态（帧率、话题、累计帧数），用于健康检查与排障。"""
        import json

        with urllib.request.urlopen(f"{self._base}/meta.json", timeout=self._timeout) as resp:
            return json.loads(resp.read().decode())

    def close(self) -> None:
        """无长连接需要释放。"""
        return None


def main(argv=None) -> int:
    """自检: python3 -m inspection_perception.capture.http_source --save 3"""
    import argparse
    import os

    ap = argparse.ArgumentParser(description="HTTP 取图客户端自检")
    ap.add_argument("--url", default="http://127.0.0.1:8730")
    ap.add_argument("--long-side", type=int, default=0)
    ap.add_argument("--save", type=int, default=1, help="保存几帧")
    ap.add_argument("--out-dir", default="_bootstrap/ros_probe")
    args = ap.parse_args(argv)

    src = HttpFrameSource(args.url, long_side=args.long_side or None)
    print("服务端状态:", src.meta())

    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(args.save):
        t0 = time.monotonic()
        frame = src.get_latest(timeout_s=5.0)
        if frame is None:
            print("！未取到新鲜帧")
            return 1
        rgb = frame.as_rgb()
        path = os.path.join(args.out_dir, f"http_{i:02d}.jpg")
        with open(path, "wb") as fh:
            fh.write(frame.jpeg or b"")
        print(f"{frame.describe()} 解码后={rgb.shape} 取图耗时={1000*(time.monotonic()-t0):.0f}ms -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
