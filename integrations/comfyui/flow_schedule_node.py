"""
ComfyUI custom node: FlowSigmaScheduleNode.

Generates a sigma tensor using the sd-flow budget algorithm. The output
plugs directly into the ``sigmas`` input of ``SamplerCustomAdvanced``
(or any node that accepts a Sigmas tensor).

Inspired by the ``KarrasScheduler`` node from ``comfy_extras.nodes_custom_sampler``.
"""

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from sd_flow.schedule import FlowSigmaSchedule


class FlowSigmaScheduleNode(io.ComfyNode):
    """
    Generates a sigma schedule using the flow budget algorithm.

    Divides the sigma range into 4 tiers (Priority, Normal, Low, Deficit)
    and uses rotating dispatch to fairly distribute steps across all noise
    levels, preventing any sigma range from being starved of compute.
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
                    "rho",
                    default=7.0,
                    min=0.0,
                    max=100.0,
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
        rho,
        budget_max,
        budget_min,
        tier_thresholds_preset="default (1.5 / 1.0 / 0.5)",
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
            rho=rho,
            budget_max=budget_max,
            budget_min=budget_min,
            tier_thresholds=tier_thresholds,
        )
        sigmas = schedule.generate_schedule()
        return io.NodeOutput(sigmas)

    get_sigmas = execute
