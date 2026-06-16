#!/usr/bin/env bash
# safe-install.sh — One-command install of sd-flow.
#
# Usage:
#   bash safe-install.sh
#
# Detects an existing ComfyUI installation and injects sd-flow as a
# custom node. Falls back to pip-only install with manual instructions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INJECT_SCRIPT="$SCRIPT_DIR/integrations/comfyui/inject.sh"

echo "╔══════════════════════════════════════════════════╗"
echo "║         sd-flow — Flow Scheduler Install        ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Check Python ────────────────────────────────────────────────────────
echo "🔍 Checking Python..."
PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        ver=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
        if [ -n "$ver" ] && python -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
            PYTHON="$candidate"
            echo "  ✓ Found Python $ver"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "  ✗ Python >=3.10 is required."
    exit 1
fi

# ── Check PyTorch ───────────────────────────────────────────────────────
echo "🔍 Checking PyTorch..."
if "$PYTHON" -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')" 2>/dev/null; then
    true
else
    echo "  ⚠  PyTorch not found. sd-flow requires PyTorch."
    echo "     Install it from https://pytorch.org/get-started/locally/"
    echo "     then re-run this installer."
fi

# ── comfyui detection + injection ──────────────────────────────────────
echo ""
echo "🔌 Checking for ComfyUI installation..."
if [ -x "$INJECT_SCRIPT" ]; then
    if bash "$INJECT_SCRIPT"; then
        echo ""
        echo "✅ sd-flow installed successfully!"
        echo "   Restart ComfyUI to use FlowSigmaSchedule and FlowSampler nodes."
        exit 0
    fi
else
    echo "  ⚠  Inject script not found at: $INJECT_SCRIPT"
fi

# ── Fallback: pip install only ──────────────────────────────────────────
echo ""
echo "📦 Installing sd-flow Python package (standalone mode)..."
cd "$SCRIPT_DIR"
"$PYTHON" -m pip install -e . 2>&1 | tail -3

echo ""
echo "✅ sd-flow installed (pip package only)."
echo ""
echo "   To integrate with ComfyUI, run:"
echo "     bash integrations/comfyui/inject.sh /path/to/ComfyUI"
echo ""
echo "   To verify:"
echo "     $PYTHON -c 'from sd_flow import FlowSigmaSchedule; print(\"OK\")'"
