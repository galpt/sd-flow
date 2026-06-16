"""
sd-flow: Flow Scheduler custom node pack for ComfyUI.

At load time, this module monkey-patches ComfyUI's sampler registry so that
"flow_heun" and "flow_euler" appear in EVERY built-in KSampler's dropdown
list -- no workflow changes needed.

Also registers standalone nodes (FlowSigmaSchedule, FlowSampler) for
use with the SamplerCustomAdvanced workflow.
"""

import logging
import os
import sys

_sd_flow_logger = logging.getLogger("sd-flow")

# ── Monkey-patch ComfyUI's sampler registry ─────────────────────────────
try:
    from sd_flow.sampler import sample_flow_heun, sample_flow_euler

    from comfy.k_diffusion import sampling as _k_sampling
    import comfy.samplers as _samplers

    # 1. Inject sampler functions into k_diffusion_sampling module
    _k_sampling.sample_flow_heun = sample_flow_heun
    _k_sampling.sample_flow_euler = sample_flow_euler
    _sd_flow_logger.info("Injected sample_flow_heun / sample_flow_euler into k_diffusion_sampling")

    # 2. Register names in KSAMPLER_NAMES (populates KSampler dropdown)
    for _name in ("flow_heun", "flow_euler"):
        if _name not in _samplers.KSAMPLER_NAMES:
            _samplers.KSAMPLER_NAMES.append(_name)

    # 3. SAMPLER_NAMES is a static copy built at import time:
    #    KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]
    #    Rebuild it so the UI sees our new entries.
    _extra_non_k = [n for n in _samplers.SAMPLER_NAMES
                    if n not in _samplers.KSAMPLER_NAMES]
    _samplers.SAMPLER_NAMES = list(_samplers.KSAMPLER_NAMES) + _extra_non_k

    _sd_flow_logger.info("Registered flow_heun / flow_euler in KSampler dropdowns")
except Exception as _exc:
    _sd_flow_logger.warning("Failed to patch KSampler dropdowns: %s", _exc)

# ── Standalone nodes ─────────────────────────────────────────────────────
# Use absolute imports because ComfyUI loads __init__.py with a mangled
# module name, which breaks Python's relative import resolution.

_node_dir = os.path.dirname(os.path.abspath(__file__))
if _node_dir not in sys.path:
    sys.path.insert(0, _node_dir)

from flow_schedule_node import FlowSigmaScheduleNode  # noqa: E402
from flow_sampler_node import FlowSamplerNode          # noqa: E402

NODE_CLASS_MAPPINGS = {
    "FlowSigmaSchedule": FlowSigmaScheduleNode,
    "FlowSampler": FlowSamplerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlowSigmaSchedule": "Flow Sigma Schedule",
    "FlowSampler": "Flow Sampler",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
