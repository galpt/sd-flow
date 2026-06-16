"""
Flow-based sigma schedule generator.

Adapts the scx_flow budget-and-tier concept to diffusion sampling.
The schedule uses linear sigma spacing (matching ComfyUI's built-in
"Normal" scheduler) to produce a clean, monotonic sigma sequence
without the noise-range concentration artifacts of polynomial
schedules.  Per-step tier indices are computed from budget
accumulation and stored in ``step_tiers`` for the adaptive sampler.
"""

import torch

from .budget import BudgetAccumulator
from .tiers import Tier, segment_sigma_range


class FlowSigmaSchedule:
    """
    A sigma schedule based on the flow budget algorithm.

    Uses linear sigma spacing (matching ComfyUI's "Normal" scheduler),
    which avoids the low-noise concentration bias of polynomial schedules
    (e.g. Karras with ρ=7) that can cause pixelation in low-step-count
    regimes.  Per-step tier indices are computed from budget accumulation
    and stored in ``self.step_tiers`` for the adaptive sampler.  Each
    viable tier is guaranteed at least 1 correction step so that no
    sigma range is entirely starved.

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
        rho: float = 7.0,
        budget_max: float = 2.0,
        budget_min: float = -0.5,
        tier_thresholds: tuple = (1.5, 1.0, 0.5),
    ):
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
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
        Generate the flow-based sigma schedule.

        The schedule uses linear sigma spacing (matching ComfyUI's
        "Normal" scheduler) to avoid the low-noise pixelation
        artifacts of polynomial schedules.  Per-step tier indices
        are computed from budget accumulation and stored in
        ``self.step_tiers``.

        Each viable sigma tier is guaranteed at least 1 correction
        step so that no noise range is entirely starved.

        Returns:
            torch.Tensor of shape ``(num_steps + 1,)``:
            ``[sigma_max, ..., sigma_min, 0]``
        """
        num = self.num_steps
        smin = self.sigma_min
        smax = self.sigma_max

        # --- 1. base schedule — linear in sigma (like ComfyUI "Normal") ---
        base = torch.linspace(smax, smin, num, dtype=torch.float32)

        # --- 2. append trailing 0 ---
        sigmas = torch.cat([base, torch.zeros(1)])

        # --- 3. compute per-step tiers from budget accumulation ---
        _, self.step_tiers = self._budget_for_base_schedule(base)

        # --- 4. ensure each viable tier has at least 1 correction step ---
        segments = segment_sigma_range(smin, smax)
        for ti in range(4):
            if ti not in self.step_tiers:
                seg_lo = segments[ti][1]
                seg_hi = segments[ti][2]
                if seg_hi - seg_lo > 1e-6:
                    # Demote the highest-ti step that has budget
                    # above this tier
                    for i in range(len(self.step_tiers) - 1, -1, -1):
                        if self.step_tiers[i] < ti:
                            self.step_tiers[i] = ti
                            break

        return sigmas
