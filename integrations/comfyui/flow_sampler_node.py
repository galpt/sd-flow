"""
ComfyUI custom node: FlowSamplerNode.

Creates a KSAMPLER wrapping the flow ODE solver. The output plugs
directly into the ``sampler`` input of ``SamplerCustomAdvanced``.

The default "adapt" mode uses DDIM for high-budget steps (PRIORITY/NORMAL)
and Euler Ancestral for low-budget steps (LOW/DEFICIT), optimising for
quality at 5-10 steps.
"""

from comfy_api.latest import io
import comfy.samplers
from sd_flow.sampler import sample_flow, sample_flow_heun, sample_flow_euler


class FlowSamplerNode(io.ComfyNode):
    """
    Creates a Flow Sampler that uses the adaptive solver.

    Default "adapt" mode: PRIORITY/NORMAL steps use DDIM (deterministic,
    high quality at low step count), LOW/DEFICIT steps use Euler Ancestral
    (adds variety).  Fall back to "heun" or "euler" for fixed-mode.
    Works with any sigma schedule.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FlowSampler",
            display_name="Flow Sampler",
            category="model/sampling/samplers",
            inputs=[
                io.Combo.Input("solver", options=["adapt", "heun", "euler"]),
                io.Float.Input("s_churn", default=0.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
                io.Float.Input("s_tmin", default=0.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
                io.Float.Input("s_tmax", default=9999.0, min=0.0, max=10000.0, step=0.01, round=False, advanced=True),
                io.Float.Input("s_noise", default=1.0, min=0.0, max=100.0, step=0.01, round=False, advanced=True),
            ],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls, solver, s_churn, s_tmin, s_tmax, s_noise) -> io.NodeOutput:
        _fn_map = {"adapt": sample_flow, "heun": sample_flow_heun, "euler": sample_flow_euler}
        sampler_fn = _fn_map.get(solver, sample_flow)
        custom_sampler = comfy.samplers.KSAMPLER(
            sampler_fn,
            extra_options={"s_churn": s_churn, "s_tmin": s_tmin, "s_tmax": s_tmax, "s_noise": s_noise},
        )
        return io.NodeOutput(custom_sampler)

    get_sampler = execute
