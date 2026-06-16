"""
ComfyUI custom node: FlowSamplerNode.

Creates a KSAMPLER wrapping the flow ODE solver. The output plugs
directly into the ``sampler`` input of ``SamplerCustomAdvanced``.
"""

from comfy_api.latest import io
import comfy.samplers
from sd_flow.sampler import sample_flow_heun, sample_flow_euler


class FlowSamplerNode(io.ComfyNode):
    """
    Creates a Flow Sampler that uses the flow ODE solver.

    Supports Heun's 2nd order (default) and Euler's 1st order solvers.
    Works with any sigma schedule, not just FlowSigmaSchedule.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FlowSampler",
            display_name="Flow Sampler",
            category="model/sampling/samplers",
            inputs=[
                io.Combo.Input("solver", options=["heun", "euler"]),
                io.Float.Input(
                    "s_churn",
                    default=0.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    round=False,
                    advanced=True,
                ),
                io.Float.Input(
                    "s_tmin",
                    default=0.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    round=False,
                    advanced=True,
                ),
                io.Float.Input(
                    "s_tmax",
                    default=float("inf"),
                    min=0.0,
                    max=10000.0,
                    step=0.01,
                    round=False,
                    advanced=True,
                ),
                io.Float.Input(
                    "s_noise",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    round=False,
                    advanced=True,
                ),
            ],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls, solver, s_churn, s_tmin, s_tmax, s_noise) -> io.NodeOutput:
        # Select the sampling function based on solver choice
        sampler_fn = sample_flow_heun if solver == "heun" else sample_flow_euler

        # Create a KSAMPLER directly (bypasses name-based ksampler() lookup)
        custom_sampler = comfy.samplers.KSAMPLER(
            sampler_fn,
            extra_options={
                "s_churn": s_churn,
                "s_tmin": s_tmin,
                "s_tmax": s_tmax,
                "s_noise": s_noise,
            },
        )
        return io.NodeOutput(custom_sampler)

    get_sampler = execute
