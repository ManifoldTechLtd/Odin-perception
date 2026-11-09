"""检查任务清单的加载。

支持 YAML 和 JSON。YAML 更适合人工维护点位清单，JSON 便于上层调度系统生成，
两种都能读——但 PyYAML 是可选依赖，只用 JSON 的环境不必安装它。

清单格式（YAML）::

    version: 1
    defaults:            # 可选，所有 item 的默认值
      severity: normal
    items:
      - id: valve_01_closed
        name: 1号阀门关闭状态
        kind: yes_no
        question: Is the red valve handle in the closed position
        expect: "yes"
        roi: [0.35, 0.40, 0.65, 0.75]
        location: 1号泵房-东侧
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from .checks.ops import validate_rule
from .schemas import CheckItem


def load_check_items(path: str) -> List[CheckItem]:
    """从 YAML/JSON 文件读取任务清单。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 格式不对、字段非法、或 id 重复。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"任务清单不存在: {path}")

    data = _load_raw(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} 的顶层应是映射，实际是 {type(data).__name__}")

    raw_items = data.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError(f"{path} 缺少非空的 items 列表")

    defaults: Dict[str, Any] = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError(f"{path} 的 defaults 应是映射")

    items: List[CheckItem] = []
    seen = set()
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise ValueError(f"{path} items[{index}] 应是映射")
        merged = {**defaults, **raw}
        if "id" not in merged:
            raise ValueError(f"{path} items[{index}] 缺少 id")
        item = CheckItem.from_dict(merged)
        # rule 的合法性在加载时校（不在 schemas.from_dict 里，那会和 ops 循环导入）：
        # 拼错参数名、未知 op、map 键写错都当场报错，而不是等跑到现场才发现全是 UNKNOWN
        validate_rule(item.rule, item.id)
        # id 重复会让存证互相覆盖、历史对比错位，必须当场拦住
        if item.id in seen:
            raise ValueError(f"{path} 中 id 重复: {item.id!r}")
        seen.add(item.id)
        items.append(item)

    return items


def _load_raw(path: str) -> Any:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as exc:
            raise ValueError(
                f"读取 {path} 需要 PyYAML（uv pip install pyyaml），"
                f"或把清单改写成 .json"
            ) from exc
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    raise ValueError(f"不支持的清单格式: {path}（需要 .yaml/.yml/.json）")
