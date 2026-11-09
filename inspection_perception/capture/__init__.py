"""取图层：把各种图像来源统一成 `FrameSource`。

当前实现：
    OdinFrameSource       ROS 2（rclpy）订阅 Odin 话题
    OdinFrameSourceRos1   ROS 1（rospy）订阅 Odin 话题
    HttpFrameSource       frame_server HTTP 端点，只需标准库
    make_frame_source()   按环境自动选择上面三者

两个 ROS 实现通过 `__getattr__` 延迟导入——推理进程既没有 rclpy 也没有 rospy，
`from inspection_perception.capture import HttpFrameSource` 不应因此失败。
"""
from __future__ import annotations

from .base import Frame, FrameSource, LatestFrameStore
from .factory import describe_ros_env, detect_ros_version, make_frame_source
from .http_source import HttpFrameSource

__all__ = [
    "Frame",
    "FrameSource",
    "LatestFrameStore",
    "HttpFrameSource",
    "OdinFrameSource",
    "OdinFrameSourceRos1",
    "make_frame_source",
    "detect_ros_version",
    "describe_ros_env",
]

_LAZY = {
    "OdinFrameSource": (".ros_odin_node", "OdinFrameSource"),
    "OdinFrameSourceRos1": (".ros1_odin_node", "OdinFrameSourceRos1"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
