#!/usr/bin/env python3
"""环境自检：安装后先跑这个，确认「哪一层可用」。

用法：
  python scripts/doctor.py [--json] [--model-dir PATH]

检查项与判定：
  python       >= 3.8（NX 上 3.8 可用，开发机通常 3.10+）
  deps         运行依赖是否可导入（缺哪个写哪个）
  device       torch 是否可用、是否有 CUDA
  model        权重目录是否完整（florence2 需要 config.json/preprocessor_config.json）
  package      本包是否已安装、CLI 入口是否在 PATH
  ros          可选：ROS1/ROS2 环境是否可见（只影响 --source ros1/ros2）

退出码：0 = 至少 dummy 可用；1 = 连基础依赖都不全。
"""

import argparse
import importlib
import json
import os
import shutil
import sys

MIN_PY = (3, 8)
DEPS = ["numpy", "PIL", "yaml", "cv2", "torch", "transformers", "huggingface_hub"]
MODEL_FILES = ["config.json", "preprocessor_config.json"]
# Florence-2 走 transformers remote code，版本敏感：偏离即推理失败（详见 README）
PINNED = {"transformers": "4.46.3", "timm": "1.0.9"}


def check_python():
    ok = sys.version_info[:2] >= MIN_PY
    return {"ok": ok, "value": "%d.%d.%d" % sys.version_info[:3],
            "detail": "" if ok else "需要 >= %d.%d" % MIN_PY}


def check_deps():
    missing, versions = [], {}
    for name in DEPS:
        try:
            mod = importlib.import_module(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            missing.append("%s(%s)" % (name, type(exc).__name__))
    return {"ok": not missing, "value": versions, "detail": "缺失: " + ", ".join(missing) if missing else ""}


def check_device():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "value": None, "detail": "torch 不可用: %s" % exc}
    cuda = bool(torch.cuda.is_available())
    return {"ok": True, "value": {"cuda": cuda,
                                  "device_name": torch.cuda.get_device_name(0) if cuda else "cpu"},
            "detail": "" if cuda else "无 CUDA，将用 CPU（Florence-2 base 可用，速度较慢）"}


def check_versions():
    """检查对 Florence-2 敏感的三个包是否在约束内。"""
    problems = []
    value = {}
    for name, want in PINNED.items():
        try:
            mod = importlib.import_module(name)
            got = getattr(mod, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            problems.append("%s 缺失(%s)" % (name, type(exc).__name__))
            value[name] = "missing"
            continue
        value[name] = got
        if got != want:
            problems.append("%s=%s（需 ==%s）" % (name, got, want))
    try:
        import einops  # noqa: F401
        value["einops"] = getattr(einops, "__version__", "ok")
    except Exception:  # noqa: BLE001
        problems.append("einops 缺失（remote code 必需）")
        value["einops"] = "missing"
    return {"ok": not problems, "value": value,
            "detail": ("；".join(problems) + "。执行 pip install -r requirements.txt --force-reinstall")
                      if problems else ""}


def check_model(model_dir):
    if not model_dir or not os.path.isdir(model_dir):
        return {"ok": False, "value": model_dir,
                "detail": "权重目录不存在；执行 bash scripts/download_models.sh florence2-base-ft"}
    missing = [f for f in MODEL_FILES if not os.path.isfile(os.path.join(model_dir, f))]
    size = 0
    for root, _dirs, files in os.walk(model_dir):
        for name in files:
            try:
                size += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return {"ok": not missing, "value": {"path": os.path.abspath(model_dir),
                                         "size_mb": round(size / 1048576.0, 1)},
            "detail": ("缺少: " + ", ".join(missing)) if missing else ""}


def check_package():
    try:
        mod = importlib.import_module("inspection_perception")
        version = getattr(mod, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "value": None, "detail": "包未安装或不可导入: %s" % exc}
    cli = shutil.which("perception-run")
    # 能 import 即视为可用：源码目录 + PYTHONPATH 直接跑是合法用法，
    # CLI 不在 PATH 只说明没做 pip install -e .，不构成失败。
    return {"ok": True, "value": {"version": version, "cli": cli or "(不在 PATH)"},
            "detail": "" if cli else "CLI 未安装到 PATH；可用 `python -m inspection_perception.cli`，"
                                     "或 pip install -e . 后使用 perception-run"}


def check_ros():
    ros_version = os.environ.get("ROS_VERSION", "")
    found = []
    for name in ("rclpy", "rospy", "sensor_msgs"):
        try:
            importlib.import_module(name)
            found.append(name)
        except Exception:  # noqa: BLE001
            pass
    return {"ok": True, "value": {"ROS_VERSION": ros_version or "(未设置)", "importable": found},
            "detail": "" if found else "无 ROS 环境，仅影响 --source ros1/ros2；file/http 不受影响"}


def main():
    parser = argparse.ArgumentParser(description="odin-perception 环境自检")
    parser.add_argument("--json", action="store_true", help="输出机读 JSON")
    parser.add_argument("--model-dir", default=os.environ.get(
        "PERCEPTION_MODEL_DIR", "models/florence2-base-ft"))
    args = parser.parse_args()

    report = {
        "python": check_python(),
        "deps": check_deps(),
        "versions": check_versions(),
        "device": check_device(),
        "model": check_model(args.model_dir),
        "package": check_package(),
        "ros": check_ros(),
    }
    # dummy 只需 python+包；florence2 还需 deps+版本约束+model
    report["usable"] = {
        "dummy": report["python"]["ok"] and report["package"]["ok"],
        "florence2": (report["python"]["ok"] and report["deps"]["ok"]
                      and report["versions"]["ok"] and report["model"]["ok"]
                      and report["package"]["ok"]),
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("odin-perception 自检")
        for name, item in report.items():
            if name == "usable":
                continue
            flag = "✓" if item["ok"] else "✗"
            print("  %s %-8s %s" % (flag, name, item["value"]))
            if item["detail"]:
                print("             %s" % item["detail"])
        print("")
        print("  可用后端: dummy=%s florence2=%s" % (
            "是" if report["usable"]["dummy"] else "否",
            "是" if report["usable"]["florence2"] else "否"))
        if not report["usable"]["florence2"]:
            print("  下一步：bash scripts/setup.sh && source .venv/bin/activate && "
                  "bash scripts/download_models.sh florence2-base-ft")

    return 0 if report["usable"]["dummy"] else 1


if __name__ == "__main__":
    sys.exit(main())
