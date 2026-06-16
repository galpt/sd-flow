#!/usr/bin/env bash
# inject.sh — Inject sd-flow custom node into a ComfyUI installation.
#
# Usage:
#   bash inject.sh [path/to/ComfyUI]
#
# If ComfyUI path is omitted, common locations are auto-detected.
# The custom node directory and the sd_flow pip package are installed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET_NAME="sd-flow"

# ── Auto-detect ComfyUI ──────────────────────────────────────────────────
find_comfyui() {
    local candidates=(
        "/mnt/data/sdxl/comfy-sage/ComfyUI"
        "$HOME/ComfyUI"
        "../ComfyUI"
        "./ComfyUI"
        "/opt/ComfyUI"
        "/workspace/ComfyUI"
    )
    local path_var="${COMFYUI_PATH:-}"
    if [ -n "$path_var" ]; then
        candidates=("$path_var" "${candidates[@]}")
    fi

    for dir in "${candidates[@]}"; do
        dir="$(realpath -m "$dir" 2>/dev/null || echo "$dir")"
        if [ -d "$dir/custom_nodes" ] && [ -f "$dir/main.py" ]; then
            echo "$dir"
            return 0
        fi
    done
    return 1
}

COMFYUI_DIR="${1:-}"
if [ -z "$COMFYUI_DIR" ]; then
    COMFYUI_DIR="$(find_comfyui)" || true
fi

if [ -z "$COMFYUI_DIR" ] || [ ! -d "$COMFYUI_DIR/custom_nodes" ]; then
    echo "⚠  ComfyUI not found."
    echo ""
    echo "   You can install the sd-flow Python library manually:"
    echo "     cd '$REPO_ROOT'"
    echo "     pip install -e ."
    echo ""
    echo "   Then copy the custom node manually:"
    echo "     cp -r '$SCRIPT_DIR' '/path/to/ComfyUI/custom_nodes/$TARGET_NAME'"
    echo ""
    echo "   Or set COMFYUI_PATH and re-run:"
    echo "     COMFYUI_PATH=/path/to/ComfyUI bash '$SCRIPT_DIR/inject.sh'"
    exit 1
fi

COMFYUI_DIR="$(realpath "$COMFYUI_DIR")"
TARGET_DIR="$COMFYUI_DIR/custom_nodes/$TARGET_NAME"

echo "📦 Installing sd-flow Python package..."
cd "$REPO_ROOT"
pip install -e . 2>&1 | tail -3

echo ""
echo "🔌 Injecting custom node into: $COMFYUI_DIR"
mkdir -p "$TARGET_DIR"

# Copy the custom node files (__init__.py, flow_*.py)
cp "$SCRIPT_DIR/__init__.py"   "$TARGET_DIR/"
cp "$SCRIPT_DIR/flow_schedule_node.py" "$TARGET_DIR/"
cp "$SCRIPT_DIR/flow_sampler_node.py"  "$TARGET_DIR/"

echo "   → $TARGET_DIR/"
ls -la "$TARGET_DIR/"

echo ""
echo "✅ sd-flow injected into: $COMFYUI_DIR"
echo "   Custom node dir: custom_nodes/$TARGET_NAME"
echo ""
echo "   Restart ComfyUI to see the FlowSigmaSchedule and FlowSampler nodes"
echo "   in the node menu under model/sampling/."
echo ""
echo "   To remove: bash '$SCRIPT_DIR/remove.sh' '$COMFYUI_DIR'"
