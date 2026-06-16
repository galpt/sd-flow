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

### "Images look different from what I expected"

**That's expected.** The `flow` sampler uses an adaptive solver per step — high-budget steps get DDIM (deterministic) while low-budget steps get Euler Ancestral (adds variety). This produces a different result from fixed-solver samplers.

Try switching between tier threshold presets on the `FlowSigmaSchedule` node:
- **Default**: Balanced correction distribution
- **Aggressive**: Fewer steps get Heun correction (higher bar)
- **Gentle**: More steps get Heun correction (lower bar)

### "Images look blurry / lack detail"

The `flow` sampler uses Euler Ancestral for LOW/DEFICIT budget tiers (adds noise for variety), which may produce slightly different results compared to a purely deterministic sampler at very low step counts.

Try:
1. Increase `num_steps` (e.g., 25–30 instead of 18)
2. Use the "Gentle" tier threshold preset (lowers bar for Heun correction)
3. Switch solver to `heun` on the Flow Sampler node for full correction on every step

### "Images have artifacts / weird patterns"

All sigma values in the flow schedule are monotonically decreasing and mathematically valid. Artifacts are unlikely to come from the schedule itself.

Try:
1. Use the `heun` solver mode (all steps get Heun correction, set via the `solver` combo on the Flow Sampler node)
2. Lower `s_churn` to 0 (deterministic)
3. Ensure your model is compatible with the sigma range

### "Generation is slower than expected"

The adaptive `flow` sampler uses fewer model evaluations than full Heun (the exact number depends on your tier threshold preset). Use the `euler` solver mode for maximum speed, `heun` for maximum quality. The `adapt` mode sits between Euler and Heun.

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
sampler = FlowSampler(solver="flow")
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
| No dedicated DDIM/DPM++ sampler in the flow package | Use the built-in DDIM/DPM++ samplers with `sampler=flow` — the adaptive solver works with any sampler choice |
| No training-time support | v0.1 is inference-only |
| No A1111 automated install | Manual pip install + custom script |
| No precomputed schedule profiles | Generate schedules programmatically |

## Getting Help

- **GitHub Issues**: [https://github.com/galpt/sd-flow/issues](https://github.com/galpt/sd-flow/issues)
- **ComfyUI discussion**: Check [r/StableDiffusion](https://reddit.com/r/StableDiffusion) for community help
- **Debug logging**: Run your UI with verbose logging to capture error details
