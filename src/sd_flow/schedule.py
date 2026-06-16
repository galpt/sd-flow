"""
Flow-based sigma schedule generator.

Adapts the scx_flow budget-and-tier concept to sigma spacing using
BUDGET-WEIGHTED STEP SELECTION.  A dense reference schedule (1000 steps)
is generated, budgets are computed per transition, and the desired
number of steps are sampled from the cumulative budget distribution.
This naturally concentrates steps where the noise dynamics change
fastest (high sigma change = high budget), producing a schedule that
adapts to the model's sigma range without any heuristics or discrete
tier boundaries.

Per-step tier indices are stored in ``step_tiers`` for the adaptive
sampler.
"""

import torch

from .budget import BudgetAccumulator
from .tiers import segment_sigma_range


class FlowSigmaSchedule:
    """
    A sigma schedule based on the flow budget algorithm.

    Uses budget-weighted step selection to concentrate steps where
    the noise dynamics change fastest.  The schedule is smooth,
    monotonic, and free of boundary artifacts.

    The schedule is a ``torch.Tensor`` of shape ``(num_steps + 1,)``
    with values descending from ``sigma_max`` to ``0``, compatible
    with ComfyUI's ``SamplerCustomAdvanced`` and k-diffusion sampler
    signatures.
    """

    # Number of dense reference points for budget computation.
    # Higher = smoother importance distribution.
    _DENSE_STEPS = 1000

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
        Generate the sigma schedule using budget-weighted step selection.

        Algorithm:
          1. Build a dense linear reference (1000 points).
          2. Compute budget for each transition, normalize to importance.
          3. Interpolate the desired number of steps from the cumulative
             importance distribution.
          4. Compute per-step tier labels from budget accumulation.
          5. Append trailing 0.

        Returns:
            torch.Tensor of shape ``(num_steps + 1,)``:
            ``[sigma_max, ..., sigma_min, 0]``
        """
        num = self.num_steps
        smin = self.sigma_min
        smax = self.sigma_max

        if num < 2:
            return torch.tensor([smax, 0.0], dtype=torch.float32)

        # --- 1. dense linear reference for budget computation ---
        dense = torch.linspace(smax, smin, self._DENSE_STEPS, dtype=torch.float64)

        # --- 2. compute budgets on dense grid ---
        budgets_64, _ = self._budget_for_base_schedule(dense)

        # Normalise budgets to [0, 1] as importance weights.
        # Shift by the minimum so the lowest-budget step still has
        # some weight, preventing collapsed spacing.
        imp = torch.tensor(budgets_64, dtype=torch.float64)
        imp = imp - float(imp.min())
        imp = imp + 1e-8  # avoid zero
        imp = imp / float(imp.sum())

        # --- 3. cumulative importance → warped positions ---
        cum = torch.cat([torch.zeros(1, dtype=torch.float64), imp.cumsum(0)])
        # cum[n] = 1.0 (within float64 precision)

        # Sample num step positions from cumulative distribution.
        # We generate num points (indexes 0..num-1).  Position 0 = smax,
        # position num-1 = smin.  Then we append 0.
        targets = torch.linspace(0.0, 1.0, num, dtype=torch.float64)

        # For each target cumulative importance, find the sigma
        # position via linear interpolation.
        sigma_positions = torch.zeros(num, dtype=torch.float64)
        sigma_positions[0] = smax
        sigma_positions[-1] = smin
        # Interpolate interior points
        for i in range(1, num - 1):
            t = targets[i]
            # Binary search would be faster, but for 1000 points linear
            # search is fine for this one-time computation.
            idx = (cum < t).sum().item()
            idx = max(0, min(idx, len(dense) - 2))
            # Linear interpolation within the bin
            lo_cum = cum[idx].item()
            hi_cum = cum[idx + 1].item()
            lo_sig = dense[idx].item()
            hi_sig = dense[idx + 1].item()
            frac = (t - lo_cum) / (hi_cum - lo_cum) if hi_cum > lo_cum else 0.0
            sigma_positions[i] = lo_sig + frac * (hi_sig - lo_sig)

        # Convert to float32 and force bounds
        sigmas = sigma_positions.to(torch.float32)
        sigmas[0] = smax
        sigmas[-1] = smin

        # --- 4. compute per-step tiers from budget accumulation ---
        base_for_tiers = sigmas[:-1]  # exclude trailing (will add below)
        _, self.step_tiers = self._budget_for_base_schedule(base_for_tiers)

        # Ensure every viable tier has at least 1 correction step
        segments = segment_sigma_range(smin, smax)
        for ti in range(4):
            if ti not in self.step_tiers:
                seg_lo = segments[ti][1]
                seg_hi = segments[ti][2]
                if seg_hi - seg_lo > 1e-6:
                    for j in range(len(self.step_tiers) - 1, -1, -1):
                        if self.step_tiers[j] < ti:
                            self.step_tiers[j] = ti
                            break

        # --- 5. append trailing 0 ---
        sigmas = torch.cat([sigmas, torch.zeros(1)])

        return sigmas
