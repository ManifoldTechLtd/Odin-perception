#!/usr/bin/env bash
# 建立隔离 venv 并安装本包（不碰系统 python / conda）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
VENV="${VENV_DIR:-$ROOT/.venv}"

echo "[1/4] 解释器: $($PY -V)"
echo "[2/4] 创建 venv: $VENV"
"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "[3/4] 安装依赖"
python -m pip install -U pip
pip install -r "$ROOT/requirements.txt"
echo "[4/4] 安装本包（editable）"
pip install -e "$ROOT"

echo ""
echo "✅ 完成。使用方式："
echo "   source $VENV/bin/activate"
echo "   perception-run --backend dummy --tasks configs/tasks/smoke.json --source file --file-path <图片或目录>"
echo "   python scripts/doctor.py          # 自检"
