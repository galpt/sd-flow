#!/usr/bin/env bash
# find-comfyui.sh — Heuristic ComfyUI installation path scanner.
#
# Scans common installation paths and reports all that look like
# a valid ComfyUI directory (has custom_nodes/ and main.py).
#
# Usage:
#   bash find-comfyui.sh          # Print all found paths
#   bash find-comfyui.sh --first  # Print only the first match

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

candidates=(
    "/mnt/data/sdxl/comfy-sage/ComfyUI"
    "$HOME/ComfyUI"
    "$(realpath "$SCRIPT_DIR/../ComfyUI" 2>/dev/null || echo "$SCRIPT_DIR/../ComfyUI")"
    "./ComfyUI"
    "/opt/ComfyUI"
    "/workspace/ComfyUI"
    "/content/ComfyUI"
    "/notebooks/ComfyUI"
    # Common ComfyUI manager default
    "$HOME/Documents/ComfyUI"
)

# Also check COMFYUI_PATH env var
if [ -n "${COMFYUI_PATH:-}" ]; then
    candidates=("$COMFYUI_PATH" "${candidates[@]}")
fi

mode="${1:-all}"
found=0

for dir in "${candidates[@]}"; do
    # Skip empty paths
    [ -z "$dir" ] && continue
    # Resolve if possible
    resolved="$(realpath -m "$dir" 2>/dev/null || echo "$dir")"
    if [ -d "$resolved/custom_nodes" ] && [ -f "$resolved/main.py" ]; then
        echo "$resolved"
        found=$((found + 1))
        if [ "$mode" = "--first" ]; then
            exit 0
        fi
    fi
done

if [ "$found" -eq 0 ]; then
    echo "No ComfyUI installation found." >&2
    exit 1
fi
