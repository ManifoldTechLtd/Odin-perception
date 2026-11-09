"""取图 HTTP 服务：把 `FrameSource` 暴露成一个极简 HTTP 端点。

**为什么需要这一层（这是本项目最重要的一处架构决策）**

ROS 2 Jazzy 的 `rclpy` 绑定在系统 Python 3.12 上，而 VLM-FO1 官方环境是
Python 3.10 + torch 2.6 + transformers 4.50。把两者塞进同一个解释器，要么
需要为 3.10 重编 ROS，要么冒着在 3.12 上跑未经验证的模型栈的风险——两种都
会在 Orin NX 的 JetPack 环境里放大成难以排查的问题。

所以取图与推理**从第一天就分成两个进程**：

    [ROS 环境 python3.12]                    [推理环境 python3.10]
    OdinFrameSource ──► frame_server  ──HTTP──►  HttpFrameSource ──► VLM
    /odin1/compressed_image                       (开发机 / 边缘服务器 / NX 本机)

这条边界同时把「边端分离」白送给我们：把 URL 从 127.0.0.1 换成 NX 的 IP，
NX 只跑取图，推理放开发机或边缘服务器，业务代码一行不改。

只用标准库 `http.server`，不引入 FastAPI/uvicorn——NX 上少一个依赖就少一个坑。
对外服务（给上层业务系统的接口）另有 `service/` 层，与本文件职责不同。
"""
from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .base import FrameSource


def _make_handler(source: FrameSource, default_quality: int, default_long_side: Optional[int]):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):  # 抑制每请求一行的默认日志
            pass

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802  (BaseHTTPRequestHandler 约定)
            parsed = urlparse(self.path)
            q = parse_qs(parsed.query)

            def qf(name: str, default: float) -> float:
                try:
                    return float(q[name][0])
                except (KeyError, ValueError):
                    return default

            if parsed.path == "/healthz":
                self._json(200, {"ok": True})
                return

            if parsed.path == "/meta.json":
                frame = source.get_latest(timeout_s=qf("timeout_s", 1.0), max_age_ms=0)
                meta = {"has_frame": frame is not None}
                if frame is not None:
                    meta.update(
                        seq=frame.seq,
                        stamp_ns=frame.stamp_ns,
                        frame_id=frame.frame_id,
                        topic=frame.topic,
                        encoding=frame.encoding,
                        width=frame.width,
                        height=frame.height,
                        age_ms=round(frame.age_ms, 1),
                    )
                if hasattr(source, "stats"):
                    meta["stats"] = source.stats()
                self._json(200, meta)
                return

            if parsed.path in ("/latest.jpg", "/latest.jpeg"):
                frame = source.get_latest(
                    timeout_s=qf("timeout_s", 2.0), max_age_ms=qf("max_age_ms", 1000.0)
                )
                if frame is None:
                    self._json(503, {"error": "no_frame", "detail": "超时未取到新鲜帧"})
                    return
                try:
                    long_side = int(qf("long_side", default_long_side or 0)) or None
                    if long_side:
                        from .decode import encode_jpeg, resize_long_side

                        body = encode_jpeg(
                            resize_long_side(frame.as_rgb(), long_side),
                            quality=int(qf("quality", default_quality)),
                        )
                    else:
                        # 直接透传 compressed 原始字节，NX 上零编解码开销
                        body = frame.as_jpeg(quality=int(qf("quality", default_quality)))
                except Exception as exc:
                    self._json(500, {"error": "encode_failed", "detail": str(exc)})
                    return

                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                # 元信息随响应头返回，客户端无需二次请求即可还原 Frame
                self.send_header("X-Frame-Seq", str(frame.seq))
                self.send_header("X-Frame-Stamp-Ns", str(frame.stamp_ns))
                self.send_header("X-Frame-Id", frame.frame_id or "")
                self.send_header("X-Frame-Topic", frame.topic or "")
                self.send_header("X-Frame-Encoding", frame.encoding or "")
                self.send_header("X-Frame-Age-Ms", f"{frame.age_ms:.1f}")
                self.end_headers()
                self.wfile.write(body)
                return

            self._json(404, {"error": "not_found", "paths": ["/latest.jpg", "/meta.json", "/healthz"]})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/annotate":
                length = int(self.headers.get("Content-Length", 0))
                if length == 0:
                    self._json(400, {"error": "empty_body"})
                    return
                body = self.rfile.read(length)
                if hasattr(source, "annotate"):
                    source.annotate(body)
                self._json(200, {"ok": True, "size": len(body)})
                return
            self._json(404, {"error": "not_found"})

    return Handler


def serve(
    source: FrameSource,
    host: str = "0.0.0.0",
    port: int = 8730,
    quality: int = 90,
    long_side: Optional[int] = None,
) -> None:
    """阻塞式运行取图 HTTP 服务。"""
    httpd = ThreadingHTTPServer((host, port), _make_handler(source, quality, long_side))
    httpd.daemon_threads = True
    print(f"[frame_server] 监听 http://{host}:{port}")
    print(f"[frame_server]   GET /latest.jpg?max_age_ms=1000&long_side=1280&quality=90")
    print(f"[frame_server]   GET /meta.json")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[frame_server] 停止")
    finally:
        httpd.server_close()


def wait_until_healthy(url: str, timeout_s: float = 30.0) -> bool:
    """轮询等待服务就绪，供启动脚本编排两个进程时使用。"""
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url.rstrip('/')}/healthz", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    return False
