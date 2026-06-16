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
| `solver` | COMBO | heun | Solver method (`heun` or `euler`) |
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

### How the KSAMPLER Wrapping Works

The `FlowSamplerNode` creates a `comfy.samplers.KSAMPLER` directly (not via the `ksampler()` factory function). This avoids patching `KSAMPLER_NAMES` or `k_diffusion_sampling.py`.

When `SamplerCustomAdvanced` calls `sampler.sample(guider, sigmas, ...)`, the `KSAMPLER.sample()` method:
1. Wraps the guider in `KSamplerX0Inpaint` (adds inpainting mask support)
2. Calls `sample_flow_heun(model_k, noise, sigmas, extra_args, callback, disable, s_churn=..., s_tmin=..., ...)`
3. The `extra_options` dict (containing `s_churn`, `s_tmin`, `s_tmax`, `s_noise`) is unpacked as keyword arguments to the sampler function

This means the stochastic churn parameters are passed directly to the ODE solver, matching the same interface as built-in k-diffusion samplers.
