"""
Flow-based sigma schedule generator.

Adapts the scx_flow budget-and-tier concept to sigma spacing.
Unlike fixed-formula schedules (Karras, Normal), the flow schedule
uses BUDGET-DRIVEN WARPING: steps in high-budget tiers (PRIORITY)
are packed closer together (more resolution), while steps in
low-budget tiers (DEFICIT) are spread wider.  The warping is
smooth and produces a clean, monotonic sigma sequence without
boundary artifacts.

Per-step tier indices are stored in ``step_tiers`` for the
adaptive sampler.
"""

import torch

from .budget import BudgetAccumulator
from .tiers import Tier, segment_sigma_range


class FlowSigmaSchedule:
    """
    A sigma schedule based on the flow budget algorithm.

    Uses budget-driven step WARPING: each step's tier determines a
    stretch factor that compresses (PRIORITY) or expands (DEFICIT)
    the sigma spacing.  This produces more resolution where the
    noise dynamics demand it, without introducing artifacts.

    The schedule is a ``torch.Tensor`` of shape ``(num_steps + 1,)``
    with values descending from ``sigma_max`` to ``0``, compatible
    with ComfyUI's ``SamplerCustomAdvanced`` and k-diffusion sampler
    signatures.
    """

    # Stretch factors per tier: <1 = denser steps, >1 = sparser steps
    _TIER_STRETCH = {0: 0.55, 1: 0.75, 2: 1.25, 3: 1.45}

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

        Algorithm:
          1. Build a linear reference schedule for budget computation.
          2. Compute per-step budget and tier for each transition.
          3. Ensure every viable tier has at least 1 correction step.
          4. For num_steps < 2, return early (no warping needed).
          5. Map each step's tier to a stretch factor.
          6. Cumulatively sum stretches to get warped step positions.
          7. Normalise positions and map to sigma values.
          8. Append the trailing 0.

        Returns:
            torch.Tensor of shape ``(num_steps + 1,)``:
            ``[sigma_max, ..., sigma_min, 0]``
        """
        num = self.num_steps
        smin = self.sigma_min
        smax = self.sigma_max

        # --- 1. linear reference schedule for budget computation ---
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

        # --- 4. for num_steps < 2, no warping is needed ---
        if num < 2:
            sigmas = torch.tensor([smax, 0.0], dtype=torch.float32)
            self.step_tiers = [3]  # deficit
            return sigmas

        # --- 5. map tiers to stretch factors ---
        stretch = torch.tensor(
            [self._TIER_STRETCH[t] for t in self.step_tiers],
            dtype=torch.float32,
        )

        # --- 6. compute warped positions ---
        # Cumulative stretch gives the relative position of each step.
        # A step with a high stretch (DEFICIT) covers more sigma range;
        # a step with a low stretch (PRIORITY) covers less.
        cum_stretch = torch.cat([
            torch.zeros(1, dtype=torch.float32),
            stretch.cumsum(0),
        ])
        total = float(cum_stretch[-1])
        if total > 0:
            positions = cum_stretch / total
        else:
            positions = torch.linspace(0, 1, num + 1, dtype=torch.float32)

        # --- 7. map positions back to sigma values ---
        sigmas = smax + positions * (smin - smax)
        sigmas[0] = smax
        sigmas[-1] = smin

        # --- 8. append trailing 0 ---
        sigmas = torch.cat([sigmas, torch.zeros(1)])

        return sigmas
