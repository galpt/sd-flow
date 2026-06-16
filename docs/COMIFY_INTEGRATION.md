# ComfyUI Integration

## How It Works

sd-flow integrates with ComfyUI through the `custom_nodes/` extension system — **no core files are patched**.

### Installation

```bash
# Auto-detect ComfyUI and inject:
bash integrations/comfyui/inject.sh

# Or specify path explicitly:
bash integrations/comfyui/inject.sh /path/to/ComfyUI
```

The inject script:
1. Copies `integrations/comfyui/` → `ComfyUI/custom_nodes/sd-flow/`
2. Installs the `sd-flow` pip package
3. Prints success message

### Removal

```bash
bash integrations/comfyui/remove.sh /path/to/ComfyUI
```

Deletes `custom_nodes/sd-flow/` — zero residue.

## Node Reference

### Flow Sigma Schedule

**Category**: `model/sampling/schedulers`
**Output**: `SIGMAS` (tensor)

Generates a sigma schedule using the flow budget algorithm. Plugs into `SamplerCustomAdvanced.sigmas`.

| Input | Type | Default | Description |
|---|---|---|---|
| `steps` | INT | 18 | Number of sampling steps |
| `sigma_max` | FLOAT | 14.614642 | Maximum noise level (advanced) |
| `sigma_min` | FLOAT | 0.0291675 | Minimum noise level (advanced) |
| `rho` | FLOAT | 7.0 | Schedule exponent within tiers (advanced) |
| `budget_max` | FLOAT | 2.0 | Maximum budget ceiling (advanced) |
| `budget_min` | FLOAT | -0.5 | Minimum budget floor (advanced) |
| `tier_thresholds_preset` | COMBO | default | Budget tier boundaries (advanced) |

> **Note**: Default `sigma_max` and `sigma_min` are set to SDXL typical values. For SD1.5, use the `KarrasScheduler` values from the built-in scheduler node.

### Flow Sampler

**Category**: `model/sampling/samplers`
**Output**: `SAMPLER`

Creates a sampler that uses the flow ODE solver. Plugs into `SamplerCustomAdvanced.sampler`.

| Input | Type | Default | Description |
|---|---|---|---|
| `solver` | COMBO | adapt | Solver mode (`adapt` = per-step tier adaptive, `heun` = all steps corrected, `euler` = all steps fast) |
| `s_churn` | FLOAT | 0.0 | Stochastic churn strength (advanced) |
| `s_tmin` | FLOAT | 0.0 | Minimum sigma for churn (advanced) |
| `s_tmax` | FLOAT | inf | Maximum sigma for churn (advanced) |
| `s_noise` | FLOAT | 1.0 | Noise multiplier for churn (advanced) |

## Example Workflow

```
[Checkpoint Loader] ──model──→ [SamplerCustomAdvanced]
                                    ↑      ↑        ↑       ↑
                               [Noise]  [Guider] [Flow Sigma] [Flow Sampler]
                                                    Schedule
```

Or use the `FlowSigmaSchedule` with **any** existing sampler for hybrid workflows.

## Compatibility

| ComfyUI Version | Status |
|---|---|
| Latest (as of 2026-06) | ✓ Tested |
| Older versions | Should work — uses `comfy_api.latest` |

The custom node uses the `ComfyExtension` API (`io.ComfyNode`) which is the current standard. If ComfyUI's internal API changes, the integration layer may need updates — the `sd_flow` library itself is unaffected.

## Technical Details

### How the KSampler Dropdown Integration Works

At load time, the `__init__.py` monkey-patches ComfyUI's sampler registry:

1. Injects `sample_flow` (plus `sample_flow_heun` and `sample_flow_euler`) into `comfy.k_diffusion.sampling`
2. Appends `"flow"`, `"flow_heun"`, `"flow_euler"` to `comfy.samplers.KSAMPLER_NAMES` and `comfy.samplers.SAMPLER_NAMES`

This means every built-in `KSampler` node's dropdown automatically includes the flow sampler — no workflow modifications needed.

### How the FlowSamplerNode Works

The `FlowSamplerNode` creates a `comfy.samplers.KSAMPLER` directly for use with `SamplerCustomAdvanced`:

```python
custom_sampler = comfy.samplers.KSAMPLER(sampler_fn, extra_options={...})
```

When `SamplerCustomAdvanced` calls `sampler.sample(guider, sigmas, ...)`:
1. `KSAMPLER.sample()` wraps the guider in `KSamplerX0Inpaint` (adds inpainting mask)
2. Calls `sample_flow(model_k, noise, sigmas, extra_args, callback, disable, s_churn=..., ...)`
3. The `extra_options` dict is unpacked as keyword arguments to the sampler function

For the `"adapt"` solver mode (default), `sample_flow` selects the solver per step based on each step's budget tier, mirroring scx_flow's variable time-slice allocation.
