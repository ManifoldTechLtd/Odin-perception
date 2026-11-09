#!/usr/bin/env bash
# 下载 Florence-2 权重到 models/florence2-base-ft（权重不入 git）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-florence2-base-ft}"

case "$MODEL" in
  florence2-base-ft) REPO="microsoft/Florence-2-base-ft" ;;
  *) echo "未知模型: $MODEL（当前仅支持 florence2-base-ft）" >&2; exit 2 ;;
esac

DEST="$ROOT/models/$MODEL"
mkdir -p "$ROOT/models"

python3 -c "import huggingface_hub" 2>/dev/null || {
  echo "缺少 huggingface_hub：先执行 bash scripts/setup.sh 并 source .venv/bin/activate" >&2
  exit 1
}

echo "仓库: $REPO"
echo "目标: $DEST"
echo "镜像: HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}（国内可设 https://hf-mirror.com）"

python3 - "$REPO" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
path = snapshot_download(repo_id=repo, local_dir=dest)
print("下载完成:", path)
PY

for f in config.json preprocessor_config.json; do
  [ -f "$DEST/$f" ] || { echo "❌ 权重不完整：缺少 $f（可重跑本脚本）" >&2; exit 1; }
done
echo "✅ 权重就绪：$(du -sh "$DEST" | cut -f1)"
