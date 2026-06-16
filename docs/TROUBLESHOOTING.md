# Troubleshooting

## Custom Node Issues

### "The Flow Sigma Schedule / Flow Sampler node doesn't appear in ComfyUI"

1. **Check injection**: Verify the files exist:
   ```bash
   ls -la /path/to/ComfyUI/custom_nodes/sd-flow/
   ```
   Should show `__init__.py`, `flow_schedule_node.py`, `flow_sampler_node.py`.

2. **Check Python package**: 
   ```bash
   pip show sd-flow
   ```
   If not installed: `pip install sd-flow` or `pip install -e /path/to/sd-flow/`

3. **Check ComfyUI console**: Look for import errors. Common issues:
   - Missing `torch` — sd-flow requires PyTorch
   - Missing `comfy_api` — custom nodes need ComfyUI's internal API
   - Python version < 3.10

4. **Restart ComfyUI**: Fully restart (not just reload UI).

### "ComfyUI fails to load after injecting sd-flow"

Most likely an import error. Check the terminal output for:

```
ImportError: No module named 'sd_flow'
```

Fix: `pip install sd-flow` before restarting ComfyUI.

If you see other errors, remove the node:

```bash
bash integrations/comfyui/remove.sh /path/to/ComfyUI
```

And check the error message. Open an issue with the full traceback.

## Quality Issues

### "Images look different from Karras schedule"

**That's expected.** The flow schedule distributes steps differently across sigma tiers. This is by design — the rotating dispatch prevents any sigma range from being starved, which produces a different (not worse or better) step distribution.

Try switching between tier threshold presets:
- **Default**: Balanced
- **Aggressive**: More steps at high noise (closer to Karras behavior)
- **Gentle**: More steps at low noise

### "Images look blurry / lack detail"

The flow schedule may allocate fewer steps to the very-low-noise range (where fine details are refined) compared to the Karras schedule with ρ=7.

Try:
1. Increase `num_steps` (e.g., 25–30 instead of 18)
2. Use the "Gentle" tier threshold preset
3. Decrease `rho` (e.g., 5.0) for more uniform step spacing

### "Images have artifacts / weird patterns"

All sigma values in the flow schedule are monotonically decreasing and mathematically valid. Artifacts are unlikely to come from the schedule itself.

Try:
1. Use the `heun` solver mode (all steps get Heun correction)
2. Lower `s_churn` to 0 (deterministic)
3. Ensure your model is compatible with the sigma range

### "Generation is slower than expected"

The adaptive `flow` sampler uses fewer model evaluations than full Heun (about 20 vs 35 for 18 steps), so it should be faster. If you're comparing against Euler, use the `euler` solver mode for maximum speed. The `adapt` mode sits between Euler and Heun in both quality and speed.

## Installation Issues

### "The inject script can't find my ComfyUI"

```bash
COMFYUI_PATH=/actual/path/to/ComfyUI bash integrations/comfyui/inject.sh
```

The script checks these locations by default:
- `/mnt/data/sdxl/comfy-sage/ComfyUI`
- `$HOME/ComfyUI`
- `../ComfyUI` (relative to sd-flow repo)
- `./ComfyUI` (current directory)
- `/opt/ComfyUI`
- `/workspace/ComfyUI`

### "I use A1111 / SD-Reforge — how do I install?"

The `sd_flow` library is a pure Python package. Install it:

```bash
pip install sd-flow
```

Then create a thin integration script that imports `FlowSigmaSchedule` and `FlowSampler`:

```python
from sd_flow import FlowSigmaSchedule, FlowSampler

schedule = FlowSigmaSchedule(num_steps=20)
sigmas = schedule.generate_schedule()
sampler = FlowSampler(solver="heun")
result = sampler.sample(denoiser_fn, latents)
```

Each UI has its own extension system — check the UI's documentation for adding custom samplers.

### "pip install fails"

Make sure you have Python ≥ 3.10 and a working PyTorch installation:

```bash
python --version
python -c "import torch; print(torch.__version__)"
```

The sd-flow package has zero pip dependencies — only the integrations require UI-specific packages (ComfyUI, etc.).

## Known Limitations (v0.1)

| Limitation | Workaround |
|---|---|
| No DDIM/DPM++ solver | Use Heun or Euler solver |
| No training-time support | v0.1 is inference-only |
| No A1111 automated install | Manual pip install + custom script |
| No precomputed schedule profiles | Generate schedules programmatically |

## Getting Help

- **GitHub Issues**: [https://github.com/galpt/sd-flow/issues](https://github.com/galpt/sd-flow/issues)
- **ComfyUI discussion**: Check [r/StableDiffusion](https://reddit.com/r/StableDiffusion) for community help
- **Debug logging**: Run your UI with verbose logging to capture error details
