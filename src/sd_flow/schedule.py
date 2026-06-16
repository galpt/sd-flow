"""
Flow-based sigma schedule generator.

Adapts the scx_flow budget-and-tier concept to sigma spacing using
budget-weighted step selection with ODE-stability enforcement.
Steps are sampled from the cumulative budget distribution, then
post-processed to ensure the step-to-step size ratio stays within
a safe bound for ODE stability.

Per-step tier indices are stored in ``step_tiers`` for the adaptive
sampler.
"""

import torch

from .budget import BudgetAccumulator
from .tiers import segment_sigma_range


class FlowSigmaSchedule:
    """
    A sigma schedule based on the flow budget algorithm.

    Uses budget-weighted step selection with smooth enforcement:
    steps are sampled from the cumulative budget distribution and
    then adjusted so consecutive steps never differ by more than
    the max allowable ratio.  This produces a schedule that
    concentrates steps in high-budget regions without causing ODE
    solver instability.

    The schedule is a ``torch.Tensor`` of shape ``(num_steps + 1,)``
    with values descending from ``sigma_max`` to ``0``, compatible
    with ComfyUI's ``SamplerCustomAdvanced`` and k-diffusion sampler
    signatures.
    """

    # Maximum allowed step-to-step ratio.  1.5x means the larger step
    # is at most 1.5x the adjacent smaller step.  1.0 = perfectly uniform.
    _MAX_STEP_RATIO = 1.5

    # Number of dense reference points for budget computation
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

    def _enforce_ratio(self, sigmas: torch.Tensor, max_ratio: float) -> torch.Tensor:
        """
        Post-process sigma values so no two consecutive steps differ
        by more than ``max_ratio``.  The last step (sigma_min->0) is
        excluded from ratio checks since it's always large.
        """
        result = sigmas.clone()
        n = len(result)
        for i in range(1, n - 1):
            prev_delta = float(result[i - 1] - result[i])
            next_delta = float(result[i] - result[i + 1])
            if prev_delta <= 0 or next_delta <= 0:
                continue
            ratio = max(prev_delta / next_delta, next_delta / prev_delta)
            if ratio > max_ratio:
                if prev_delta > next_delta:
                    max_allowed = prev_delta / max_ratio
                    new_sigma = float(result[i + 1]) + max_allowed
                    result[i] = min(new_sigma, float(result[i - 1]) - 1e-8)
                else:
                    max_allowed = prev_delta * max_ratio
                    new_sigma = float(result[i - 1]) - max_allowed
                    result[i] = max(new_sigma, float(result[i + 1]) + 1e-8)

        # Enforce bounds
        result[0] = self.sigma_max
        result[-1] = self.sigma_min
        return result

    def generate_schedule(self) -> torch.Tensor:
        """
        Generate the sigma schedule using budget-weighted step selection
        with ratio enforcement.

        Algorithm:
          1. Build a dense linear reference for budget computation.
          2. Compute budget for each transition, normalise to importance.
          3. Sample steps from the cumulative importance distribution.
          4. Enforce max step-to-step ratio for ODE stability.
          5. Compute per-step tier labels from budget accumulation.
          6. Append trailing 0.

        Returns:
            torch.Tensor of shape ``(num_steps + 1,)``:
            ``[sigma_max, ..., sigma_min, 0]``
        """
        num = self.num_steps
        smin = self.sigma_min
        smax = self.sigma_max

        if num < 2:
            self.step_tiers = [3]
            return torch.tensor([smax, 0.0], dtype=torch.float32)

        # --- 1. dense linear reference ---
        dense = torch.linspace(smax, smin, self._DENSE_STEPS, dtype=torch.float64)

        # --- 2. compute budgets on dense grid ---
        budgets_64, _ = self._budget_for_base_schedule(dense)

        # Normalise to importance weights with smoothing and clamping
        imp = torch.tensor(budgets_64, dtype=torch.float64)
        imp = imp - float(imp.min())
        imp_max = float(imp.max())
        if imp_max > 0:
            imp = imp / imp_max
        imp = imp + 1e-30  # avoid zero

        # Smooth with a Gaussian kernel
        ks = 10.0
        kr = int(ks * 3)
        xs = torch.arange(-kr, kr + 1, dtype=torch.float64)
        kernel = torch.exp(-0.5 * (xs / ks) ** 2)
        kernel = kernel / float(kernel.sum())
        imp = torch.conv1d(imp.view(1, 1, -1), kernel.view(1, 1, -1), padding='same').view(-1)

        # Clamp to limit density variation
        lo = float(imp.min())
        hi = float(imp.max())
        if hi > lo:
            imp = torch.clamp(imp, lo + (hi - lo) * 0.1, hi)

        imp = imp / float(imp.sum())  # probability distribution

        # --- 3. sample from cumulative importance ---
        cum = torch.cat([torch.zeros(1, dtype=torch.float64), imp.cumsum(0)])
        targets = torch.linspace(0.0, 1.0, num, dtype=torch.float64)

        sigma_positions = torch.zeros(num, dtype=torch.float64)
        sigma_positions[0] = smax
        sigma_positions[-1] = smin
        for i in range(1, num - 1):
            t = float(targets[i])
            idx = int((cum < t).sum().item())
            idx = max(0, min(idx, len(dense) - 2))
            lo_c = float(cum[idx])
            hi_c = float(cum[idx + 1])
            lo_s = float(dense[idx])
            hi_s = float(dense[idx + 1])
            frac = (t - lo_c) / (hi_c - lo_c) if hi_c > lo_c else 0.0
            sigma_positions[i] = lo_s + frac * (hi_s - lo_s)

        sigmas = sigma_positions.to(torch.float32)

        # --- 4. enforce ODE stability ---
        sigmas = self._enforce_ratio(sigmas, self._MAX_STEP_RATIO)

        # --- 5. compute per-step tiers ---
        base_for_tiers = sigmas[:-1]
        _, self.step_tiers = self._budget_for_base_schedule(base_for_tiers)

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

        # --- 6. append trailing 0 ---
        sigmas = torch.cat([sigmas, torch.zeros(1)])

        return sigmas
