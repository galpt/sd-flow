"""
ComfyUI custom node: FlowSigmaScheduleNode.

Generates a sigma tensor using the sd-flow budget algorithm with linear
sigma spacing (matching ComfyUI's "Normal" scheduler). The output plugs
directly into the ``sigmas`` input of ``SamplerCustomAdvanced`` (or any
node that accepts a Sigmas tensor).
"""

from comfy_api.latest import io

from sd_flow.schedule import FlowSigmaSchedule


class FlowSigmaScheduleNode(io.ComfyNode):
    """
    Generates a sigma schedule using the flow budget algorithm.

    Uses linear sigma spacing with per-step budget-tier labels for the
    adaptive solver. The sigma values are compatible with any sampler;
    the tier labels are read by the ``flow`` sampler to decide which
    steps get DDIM (high budget) vs Euler Ancestral (low budget).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FlowSigmaSchedule",
            display_name="Flow Sigma Schedule",
            category="model/sampling/schedulers",
            inputs=[
                io.Int.Input("steps", default=18, min=1, max=10000),
                io.Float.Input(
                    "sigma_max",
                    default=14.614642,
                    min=0.0,
                    max=5000.0,
                    step=0.01,
                    round=False,
                    advanced=True,
                ),
                io.Float.Input(
                    "sigma_min",
                    default=0.0291675,
                    min=0.0,
                    max=5000.0,
                    step=0.01,
                    round=False,
                    advanced=True,
                ),
                io.Float.Input(
                    "budget_max",
                    default=2.0,
                    min=0.1,
                    max=10.0,
                    step=0.1,
                    round=False,
                    advanced=True,
                ),
                io.Float.Input(
                    "budget_min",
                    default=-0.5,
                    min=-2.0,
                    max=2.0,
                    step=0.1,
                    round=False,
                    advanced=True,
                ),
                io.Combo.Input(
                    "tier_thresholds_preset",
                    options=["default", "aggressive", "gentle"],
                    advanced=True,
                ),
            ],
            outputs=[io.Sigmas.Output()],
        )

    @classmethod
    def execute(
        cls,
        steps,
        sigma_max,
        sigma_min,
        budget_max,
        budget_min,
        tier_thresholds_preset="default",
    ) -> io.NodeOutput:
        # Parse tier thresholds from preset
        preset_map = {
            "default": (1.5, 1.0, 0.5),
            "aggressive": (2.0, 1.5, 1.0),
            "gentle": (1.2, 0.8, 0.4),
        }
        tier_thresholds = preset_map.get(tier_thresholds_preset, (1.5, 1.0, 0.5))

        schedule = FlowSigmaSchedule(
            num_steps=steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            budget_max=budget_max,
            budget_min=budget_min,
            tier_thresholds=tier_thresholds,
        )
        sigmas = schedule.generate_schedule()
        return io.NodeOutput(sigmas)

    get_sigmas = execute
