#!/usr/bin/env bash
# remove.sh — Remove sd-flow custom node from a ComfyUI installation.
#
# Usage:
#   bash remove.sh [path/to/ComfyUI]
#
# If ComfyUI path is omitted, common locations are auto-detected.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_NAME="sd-flow"

# ── Auto-detect ComfyUI (same logic as inject.sh) ────────────────────────
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

TARGET_DIR=""
if [ -n "$COMFYUI_DIR" ]; then
    TARGET_DIR="$(realpath -m "$COMFYUI_DIR/custom_nodes/$TARGET_NAME" 2>/dev/null || true)"
fi

if [ -z "$TARGET_DIR" ] || [ ! -d "$TARGET_DIR" ]; then
    echo "⚠  sd-flow custom node not found in ComfyUI."
    echo "   (Looked in: ${COMFYUI_DIR:-auto-detect locations})"
    echo ""
    echo "   You can still uninstall the Python package:"
    echo "     pip uninstall sd-flow -y"
    exit 1
fi

echo "🗑  Removing sd-flow custom node: $TARGET_DIR"
rm -rf "$TARGET_DIR"
echo "   ✓ Removed"

echo ""
echo "   Optionally uninstall the Python package:"
echo "     pip uninstall sd-flow -y"
echo ""
echo "   Restart ComfyUI to complete removal."
