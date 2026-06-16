"""
sd-flow: Flow Scheduler custom node pack for ComfyUI.

Registers two nodes:
  - FlowSigmaSchedule  (category: model/sampling/schedulers)
  - FlowSampler        (category: model/sampling/samplers)

Both nodes are compatible with the SamplerCustomAdvanced workflow.
"""

from .flow_schedule_node import FlowSigmaScheduleNode
from .flow_sampler_node import FlowSamplerNode

NODE_CLASS_MAPPINGS = {
    "FlowSigmaSchedule": FlowSigmaScheduleNode,
    "FlowSampler": FlowSamplerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlowSigmaSchedule": "Flow Sigma Schedule",
    "FlowSampler": "Flow Sampler",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
