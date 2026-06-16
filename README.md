# sd-flow — A Fair-Scheduling Noise Sampler for Stable Diffusion

> Drop-in noise scheduler for ComfyUI. Budget-driven step allocation. No sigma range gets starved.

---

## What's this all about?

sd-flow adapts the **scx_flow** CPU scheduling algorithm — used in Linux kernel process scheduling — into a Stable Diffusion noise sampler.

The original scx_flow uses a **budget-driven tier system** with **rotating dispatch** to guarantee every task gets fair CPU time, no matter its priority. sd-flow applies the same logic to diffusion: instead of tasks competing for CPU, **sigma ranges compete for step allocation**. The result is a noise schedule that prevents any noise level from being under-sampled.

Think of it as the difference between a VIP line (Karras ρ=7, where low-noise steps get most of the attention) and a round-robin queue (flow schedule, where every sigma tier gets its turn).

### Will this work for me?

- ✅ **You use ComfyUI** — one-command install, zero core file patching
- ✅ **You have a working PyTorch environment** (SD WebUI, ComfyUI, etc.)
- ✅ **You're on Python 3.10+** (Linux, Windows, macOS)
- ❌ **You don't use any Stable Diffusion UI** — you can still use the `sd_flow` library directly

---

## Make it better (the easy way)

**One command:**

```bash
git clone https://github.com/galpt/sd-flow.git
bash sd-flow/safe-install.sh
```

What happens:
1. Detects your ComfyUI installation (checks common paths)
2. Copies the custom node into `custom_nodes/sd-flow/`
3. Installs the `sd_flow` Python package
4. **Restart ComfyUI** — the `flow` sampler appears in every built-in `KSampler` dropdown (just select it), and the `FlowSigmaSchedule` + `FlowSampler` nodes appear for `SamplerCustomAdvanced` workflows

You'll see something like:

```
+----------------------------------------------------+
|          sd-flow — Flow Scheduler Install         |
+----------------------------------------------------+

🔍 Checking Python...
  ✓ Found Python 3.11

🔌 Injecting custom node into: /home/user/ComfyUI
   → /home/user/ComfyUI/custom_nodes/sd-flow/

✅ sd-flow injected successfully!
   Restart ComfyUI — 'flow' appears in every KSampler dropdown,
   plus FlowSigmaSchedule/FlowSampler nodes for custom workflows.
```

### Undo

```bash
bash sd-flow/integrations/comfyui/remove.sh
# or just: rm -rf /path/to/ComfyUI/custom_nodes/sd-flow/
```

---

## Make it better (the manual way)

If the auto-detection fails or you want more control:

```bash
# 1. Install the Python library
pip install sd-flow

# 2. Inject into ComfyUI manually
bash sd-flow/integrations/comfyui/inject.sh /path/to/ComfyUI
```

Or set the path via environment variable:

```bash
COMFYUI_PATH=/path/to/ComfyUI bash sd-flow/integrations/comfyui/inject.sh
```

---

## So, what did that actually do?

### The non-boring explanation

sd-flow is two things:

**1. A sigma schedule generator** (`FlowSigmaSchedule`)

The sigma range [σ_max, σ_min] is divided into 4 tiers — Priority, Normal, Low, Deficit — each covering a quarter of the noise spectrum. A "budget" is tracked across sigma transitions: when sigma drops steeply (early steps), budget accumulates. When sigma changes slowly (late steps), budget drains.

Based on their budget, timesteps are classified into tiers. A **rotating dispatch** cycle ensures each tier gets its fair share of steps, preventing any sigma range from being starved.

**2. An adaptive ODE solver** (`FlowSampler`)

The solver quality per step is determined by the flow budget tier: high-budget steps (PRIORITY/NORMAL) get Heun's 2nd order correction, while low-budget steps (LOW/DEFICIT) use faster Euler. This mirrors scx_flow's variable time-slice allocation. The solver also appears as `flow` in every built-in KSampler dropdown — no manual wiring needed.

### What's different from Karras?

| Aspect | Karras (ρ=7) | sd-flow |
|--------|-------------|---------|
| Step distribution | Concentrated at low noise | Fair across all tiers |
| Control | Single `rho` parameter | Budget thresholds + dispatch |
| Starvation protection | None | Every tier gets ≥1 step |
| Determinism | Yes | Yes |

### Project tree

```
sd-flow/
├── src/sd_flow/                  ★ Core library
│   ├── budget.py                 Budget accumulator (scx_flow adaptation)
│   ├── tiers.py                  Tier enum + sigma range segmentation
│   ├── rotating_dispatch.py      4-phase rotating dispatch
│   ├── schedule.py               FlowSigmaSchedule generator
│   ├── sampler.py                FlowSampler (adaptive flow ODE solver)
│   └── utils.py                  Helpers (clamp, to_d, round_sigma)
├── integrations/comfyui/         ★ ComfyUI custom node
│   ├── __init__.py               NODE_CLASS_MAPPINGS registration
│   ├── flow_schedule_node.py     FlowSigmaSchedule node
│   ├── flow_sampler_node.py      FlowSampler node
│   ├── inject.sh                 Installation script
│   └── remove.sh                 Removal script
├── examples/example.py           Standalone usage example
├── tests/                        Unit tests (147 tests, all pass)
├── safe-install.sh               One-command installer
└── pyproject.toml                Zero runtime dependencies
```

---

## Something looks weird

### "The nodes don't appear in ComfyUI"

Run `bash integrations/comfyui/inject.sh` again, or check the [ComfyUI troubleshooting guide](docs/COMIFY_INTEGRATION.md). Most likely `pip install sd-flow` was missed.

### "The images look different from Karras"

That's expected — the step distribution is intentionally different. Try adjusting the **tier thresholds preset** on the FlowSigmaSchedule node. The "Aggressive" preset behaves more like Karras.

### "My A1111 / SD-Reforge isn't detected"

sd-flow's automated install targets ComfyUI only. For other UIs, install the Python library and create a thin integration:

```python
from sd_flow import FlowSigmaSchedule, FlowSampler

schedule = FlowSigmaSchedule(num_steps=20)
sigmas = schedule.generate_schedule()
sampler = FlowSampler(solver="heun")
result = sampler.sample(my_denoiser, latents)
```

Your AI agent can handle the UI-specific wiring in minutes.

### "Can I break something?"

No. sd-flow doesn't modify any core files. The worst case is a ComfyUI restart with an import error, which is fixed by removing the `custom_nodes/sd-flow/` directory.

---

## I want to customize things

### Python API

```python
from sd_flow import FlowSigmaSchedule, FlowSampler

# Custom schedule
schedule = FlowSigmaSchedule(
    num_steps=25,
    sigma_min=0.002,
    sigma_max=80.0,
    rho=5.0,
    budget_max=3.0,
    tier_thresholds=(2.0, 1.5, 1.0),  # "Aggressive" preset
)
sigmas = schedule.generate_schedule()

# Custom sampler (default: adaptive "flow" solver)
sampler = FlowSampler(solver="flow", s_churn=2.0, s_noise=1.5)
result = sampler.sample(denoiser_fn, latents)
```

### CLI

```bash
# Run the standalone example
PYTHONPATH=src python examples/example.py

# Run tests
bash tests/run_all.sh
```

### Parameters

| Parameter | Default | What it does |
|---|---|---|
| `num_steps` | 18 | Total sampling steps |
| `sigma_min` | 0.002 | Lowest noise level |
| `sigma_max` | 80.0 | Highest noise level |
| `rho` | 7.0 | Step spacing within tiers |
| `budget_max` | 2.0 | Max budget ceiling |
| `budget_min` | -0.5 | Min budget floor |

---

## Inspired by

- [scx_flow](https://github.com/sched-ext/scx/tree/main/scheds/experimental/scx_flow) — the original budget-driven CPU scheduler. A beautiful piece of kernel engineering: zero heuristics, starvation-free by construction, fully deterministic.
- [EDM / Karras scheduler](https://github.com/NVlabs/edm) — the mathematical foundation for noise scheduling in modern diffusion models.
- [cachy-screen-enhancer](https://github.com/galpt/cachy-screen-enhancer) — this project's install-flow template (one-command, self-bootstrapping, clear undo).

sd-flow is **not** a port of scx_flow's C/Rust code. It's a domain adaptation of the scheduling philosophy — the same rotating-dispatch fairness guarantee, applied to noise level allocation instead of CPU time.

---

## License

BSD 3-Clause. In plain English:

- **You can** use this in commercial projects, modify it, distribute it
- **You must** keep the copyright notice
- **You cannot** use the authors' names to promote your products without permission
- **No warranty** — this is provided as-is

Full text: [LICENSE](LICENSE)
