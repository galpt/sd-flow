"""
Flow-based sigma schedule generator.

Adapts the scx_flow budget-and-tier concept to solver step correction.
The sigma schedule uses clean linear spacing (same as ComfyUI's "Normal"
scheduler).  Per-step tier indices are computed from budget accumulation
and stored in ``step_tiers`` for the adaptive sampler, which is where the
flow algorithm's budget/tier/dispatch logic takes effect.
"""

import torch

from .budget import BudgetAccumulator
from .tiers import segment_sigma_range


class FlowSigmaSchedule:
    """
    A sigma schedule for the flow sampler.

    Uses clean linear sigma spacing (same as ComfyUI's "Normal" scheduler)
    to avoid the artifacts associated with non-uniform spacing.  Per-step
    tier indices are computed from budget accumulation and stored in
    ``self.step_tiers``.  These tiers are consumed by the adaptive
    ``flow`` sampler to decide which steps get Heun correction vs Euler.

    The schedule is a ``torch.Tensor`` of shape ``(num_steps + 1,)``
    with values descending from ``sigma_max`` to ``0``, compatible
    with ComfyUI's ``SamplerCustomAdvanced`` and k-diffusion sampler
    signatures.
    """

    def __init__(
        self,
        num_steps: int = 18,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        budget_max: float = 2.0,
        budget_min: float = -0.5,
        tier_thresholds: tuple = (1.5, 1.0, 0.5),
    ):
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.budget_max = budget_max
        self.budget_min = budget_min
        self.tier_thresholds = tier_thresholds

    def _budget_for_base_schedule(
        self, base_sigmas: torch.Tensor
    ) -> tuple[list[float], list[int]]:
        """Accumulate budgets across a reference schedule and return per-step
        budget values and tier indices (0=priority ... 3=deficit)."""
        accumulator = BudgetAccumulator(
            budget_max=self.budget_max,
            budget_min=self.budget_min,
            tier_thresholds=self.tier_thresholds,
        )
        tier_map = {'priority': 0, 'normal': 1, 'low': 2, 'deficit': 3}
        budgets: list[float] = []
        tiers: list[int] = []
        for i in range(len(base_sigmas) - 1):
            delta = base_sigmas[i] - base_sigmas[i + 1]
            budget = accumulator.accumulate(delta, base_sigmas[i], self.sigma_max)
            tier_str = accumulator.classify_tier(budget)
            budgets.append(budget)
            tiers.append(tier_map[tier_str])
        return budgets, tiers

    def generate_schedule(self) -> torch.Tensor:
        """
        Generate the sigma schedule.

        The schedule uses linear sigma spacing (matching ComfyUI's
        "Normal" scheduler).  Per-step tier indices are computed
        from budget accumulation and stored in ``self.step_tiers``
        for the adaptive sampler.

        Returns:
            torch.Tensor of shape ``(num_steps + 1,)``:
            ``[sigma_max, ..., sigma_min, 0]``
        """
        num = self.num_steps
        smin = self.sigma_min
        smax = self.sigma_max

        # --- 1. linear sigma schedule ---
        base = torch.linspace(smax, smin, num, dtype=torch.float32)

        # --- 2. compute per-step tiers from budget accumulation ---
        _, self.step_tiers = self._budget_for_base_schedule(base)

        # --- 3. ensure every viable tier has at least 1 correction step ---
        segments = segment_sigma_range(smin, smax)
        for ti in range(4):
            if ti not in self.step_tiers:
                seg_lo = segments[ti][1]
                seg_hi = segments[ti][2]
                if seg_hi - seg_lo > 1e-6:
                    for i in range(len(self.step_tiers) - 1, -1, -1):
                        if self.step_tiers[i] < ti:
                            self.step_tiers[i] = ti
                            break

        # --- 4. append trailing 0 ---
        sigmas = torch.cat([base, torch.zeros(1)])

        return sigmas
