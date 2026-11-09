"""取图源工厂：按当前环境自动选择 ROS 1 / ROS 2 / HTTP 实现。

因为 NX 端到底是 ROS 1 还是 ROS 2 尚未实测确认（`go2_start_roscore.sh` 指向
ROS 1，但脚本命名也可能只是沿用旧习惯），这里做成运行时探测——
探测到什么就用什么，不把猜测写死进代码。

判定顺序：
    1. 显式指定 backend 参数     → 按指定的来
    2. 环境变量 ROS_VERSION      → ROS 官方 setup.bash 会导出，最可靠
    3. 能 import rclpy           → ROS 2
    4. 能 import rospy           → ROS 1
    5. 都不行                    → 明确报错，并给出该 source 哪个 setup.bash
"""
from __future__ import annotations

import importlib.util
import os
from typing import Optional


def detect_ros_version() -> Optional[int]:
    """返回 1 / 2 / None。"""
    env = os.environ.get("ROS_VERSION", "").strip()
    if env in ("1", "2"):
        return int(env)
    if importlib.util.find_spec("rclpy") is not None:
        return 2
    if importlib.util.find_spec("rospy") is not None:
        return 1
    return None


def describe_ros_env() -> str:
    """给出当前 ROS 环境的可读描述，用于日志与报错信息。"""
    return (
        f"ROS_VERSION={os.environ.get('ROS_VERSION', '<未设置>')} "
        f"ROS_DISTRO={os.environ.get('ROS_DISTRO', '<未设置>')} "
        f"rclpy={'有' if importlib.util.find_spec('rclpy') else '无'} "
        f"rospy={'有' if importlib.util.find_spec('rospy') else '无'}"
    )


def make_frame_source(
    backend: str = "auto",
    raw_topic: str = "/odin1/image",
    compressed_topic: str = "/odin1/image/compressed",
    prefer: str = "compressed",
    long_side: Optional[int] = None,
    http_url: str = "http://127.0.0.1:8730",
    file_path: Optional[str] = None,
):
    """构造取图源。

    Args:
        backend: "auto" | "ros1" | "ros2" | "http" | "file"。
            http 用于推理进程（它没有 ROS 依赖），见 frame_server 的说明；
            file 用于离线回归。
        file_path: backend="file" 时的图片文件或目录。
        其余参数含义见各实现的 docstring。

    Returns:
        实现了 `FrameSource` 协议的对象。
    """
    if backend == "file":
        from .file_source import FileFrameSource

        if not file_path:
            raise ValueError("backend='file' 需要提供 file_path")
        return FileFrameSource(file_path)

    if backend == "http":
        from .http_source import HttpFrameSource

        return HttpFrameSource(http_url, long_side=long_side)

    if backend == "auto":
        version = detect_ros_version()
        if version is None:
            raise RuntimeError(
                "没有检测到 ROS 环境。当前: " + describe_ros_env() + "\n"
                "  ROS 2: source /opt/ros/<distro>/setup.bash\n"
                "  ROS 1: source /opt/ros/noetic/setup.bash\n"
                "  推理进程请显式用 backend='http'，不需要 ROS。"
            )
        backend = f"ros{version}"

    common = dict(
        raw_topic=raw_topic,
        compressed_topic=compressed_topic,
        prefer=prefer,
        long_side=long_side,
    )

    if backend == "ros2":
        from .ros_odin_node import OdinFrameSource

        return OdinFrameSource(**common)

    if backend == "ros1":
        from .ros1_odin_node import OdinFrameSourceRos1

        return OdinFrameSourceRos1(**common)

    raise ValueError(f"未知 backend: {backend!r}（可选 auto/ros1/ros2/http/file）")
