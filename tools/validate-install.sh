#!/usr/bin/env bash
# validate-install.sh — Verify sd-flow installation.
#
# Checks:
#   1. sd_flow Python package is importable
#   2. FlowSigmaSchedule generates a valid schedule
#   3. If COMFYUI_PATH is set or detected, verify custom node exists
#
# Usage:
#   bash validate-install.sh

set -euo pipefail

errors=0

echo "╔══════════════════════════════════════════════════╗"
echo "║       sd-flow — Installation Validation         ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Which Python to use (can be overridden via PYTHON env var)
PY_CMD="${PYTHON:-python}"

# ── Check 1: Python package ─────────────────────────────────────────────
echo "🔍 1. Python package (using: $PY_CMD)..."
if "$PY_CMD" -c "from sd_flow import FlowSigmaSchedule; print('     ✓ sd_flow imported successfully')" 2>&1; then
    true
else
    echo "     ✗ sd_flow module not importable (tried: $PY_CMD)"
    errors=$((errors + 1))
fi

# ── Check 2: Schedule generation ────────────────────────────────────────
echo "🔍 2. Schedule generation..."
if "$PY_CMD" -c "
from sd_flow import FlowSigmaSchedule
s = FlowSigmaSchedule(num_steps=10)
sigmas = s.generate_schedule()
assert len(sigmas) == 11, f'Expected 11 sigmas, got {len(sigmas)}'
assert sigmas[0] > sigmas[-2], 'Should be decreasing'
assert sigmas[-1] == 0.0, 'Last should be 0'
print('     ✓ FlowSigmaSchedule generates valid schedule')
print(f'       Shape: {list(sigmas.shape)}, first={float(sigmas[0]):.2f}, last={float(sigmas[-1])}')
" 2>&1; then
    true
else
    echo "     ✗ Schedule generation failed"
    errors=$((errors + 1))
fi

# ── Check 3: ComfyUI custom node ────────────────────────────────────────
echo "🔍 3. ComfyUI custom node..."
comfyui_path="$(bash "$(dirname "$0")/find-comfyui.sh" --first 2>/dev/null || true)"
if [ -n "$comfyui_path" ]; then
    node_dir="$comfyui_path/custom_nodes/sd-flow"
    if [ -d "$node_dir" ]; then
        echo "     ✓ Custom node found at: $node_dir"
        for f in __init__.py flow_schedule_node.py flow_sampler_node.py; do
            if [ -f "$node_dir/$f" ]; then
                echo "       ✓ $f"
            else
                echo "       ✗ Missing: $f"
                errors=$((errors + 1))
            fi
        done
    else
        echo "     ⚠  Custom node not installed."
        echo "       Run: bash $(dirname "$0")/../integrations/comfyui/inject.sh"
    fi
else
    echo "     ⚠  ComfyUI installation not detected."
    echo "       (This is fine if you're using sd-flow standalone.)"
fi

# ── Summary ─────────────────────────────────────────────────────────────
echo ""
if [ "$errors" -eq 0 ]; then
    echo "✅ All checks passed."
else
    echo "⚠  $errors check(s) failed. See messages above."
    exit 1
fi
