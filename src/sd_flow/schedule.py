"""
Flow-based sigma schedule generator.

Adapts the scx_flow rotating-dispatch concept to diffusion sampling:
instead of reordering steps (which would break ODE monotonicity), the
schedule uses budget/tier classification to determine how many steps are
allocated to each sigma region, with rotating dispatch ensuring no
region is starved.
"""

import torch

from .budget import BudgetAccumulator
from .tiers import Tier, segment_sigma_range
from .rotating_dispatch import DispatchRotator


def _karras_in_range(n_steps: int, lo: float, hi: float, rho: float) -> torch.Tensor:
    """Generate ``n_steps`` Karras-polynomial sigma values in ``(lo, hi]``."""
    if n_steps <= 0:
        return torch.tensor([], dtype=torch.float64)
    ramp = torch.linspace(0, 1, n_steps)
    lo_inv = lo ** (1 / rho)
    hi_inv = hi ** (1 / rho)
    return (hi_inv + ramp * (lo_inv - hi_inv)) ** rho


class FlowSigmaSchedule:
    """
    A sigma schedule based on the flow budget algorithm.

    Divides [sigma_min, sigma_max] into 4 tier regions, uses budget-driven
    step allocation, and applies rotating dispatch to fairly distribute steps
    across all noise levels.

    Unlike the Karras schedule (which concentrates steps near low noise via
    rho=7), the flow schedule ensures every sigma tier receives adequate
    coverage, preventing any noise range from being starved.

    The schedule is a ``torch.Tensor`` of shape ``(num_steps + 1,)`` with
    values descending from ``sigma_max`` to ``0``, compatible with ComfyUI's
    ``SamplerCustomAdvanced`` and k-diffusion sampler signatures.
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
    ) -> tuple[list[float], list[str]]:
        """Accumulate budgets across a reference schedule and return per-step
        budget values and tier labels."""
        accumulator = BudgetAccumulator(
            budget_max=self.budget_max,
            budget_min=self.budget_min,
            tier_thresholds=self.tier_thresholds,
        )
        budgets: list[float] = []
        tiers: list[str] = []
        for i in range(len(base_sigmas) - 1):
            delta = base_sigmas[i] - base_sigmas[i + 1]
            budget = accumulator.accumulate(delta, base_sigmas[i], self.sigma_max)
            tier = accumulator.classify_tier(budget)
            budgets.append(budget)
            tiers.append(tier)
        return budgets, tiers

    def generate_schedule(self) -> torch.Tensor:
        """
        Generate the flow-based sigma schedule.

        Algorithm:

        1. Segment the sigma range into 4 tier regions.
        2. Build a reference Karras schedule with ``num_steps + 1`` points
           (so there are exactly ``num_steps`` transitions).
        3. Compute budget and tier label for each transition.
        4. Count how many reference steps fall into each tier.
        5. Run rotating dispatch to re-distribute step counts, ensuring
           each tier with a viable sigma range gets at least 1 step.
        6. Generate Karras-polynomial steps within each tier's sigma range.
        7. Concatenate, enforce monotonic decreasing order, and append 0.

        Returns:
            torch.Tensor of shape ``(num_steps + 1,)``:
            ``[sigma_max, ..., sigma_min, 0]``
        """
        num = self.num_steps

        # --- 1. segment sigma range ---
        segments: list[tuple[Tier, float, float]] = segment_sigma_range(
            self.sigma_min, self.sigma_max
        )

        # --- 2. reference schedule (Karras, num_steps + 1 points) ---
        ramp = torch.linspace(0, 1, num)
        min_inv = self.sigma_min ** (1 / self.rho)
        max_inv = self.sigma_max ** (1 / self.rho)
        base = (max_inv + ramp * (min_inv - max_inv)) ** self.rho
        base = torch.cat([base, torch.zeros(1)])  # <-- num_steps + 1 points

        # --- 3. budget / tier for each transition ---
        _, tiers = self._budget_for_base_schedule(base)

        tier_map = {'priority': 0, 'normal': 1, 'low': 2, 'deficit': 3}
        tier_idx = [tier_map[t] for t in tiers]  # length = num

        # --- 4. initial step counts per tier ---
        init_counts = [tier_idx.count(t) for t in range(4)]

        # --- 5. rotating dispatch re-weighting ---
        # Final counts: each tier gets at least 1 if it has a viable range.
        final_counts = [0, 0, 0, 0]

        # Guarantee each tier at least 1 step (if its range > 0)
        for ti in range(4):
            seg_lo = segments[ti][1]
            seg_hi = segments[ti][2]
            if seg_hi - seg_lo > 1e-6:
                final_counts[ti] = 1

        already_guaranteed = sum(final_counts)
        remaining_to_assign = num - already_guaranteed

        if remaining_to_assign > 0:
            # Rotating dispatch: distribute remaining steps
            rotator = DispatchRotator(n_tiers=4)
            # Each tier's "deserved" additional steps proportional to initial count
            extra_deserved = [
                max(0, init_counts[ti] - 1) for ti in range(4)
            ]
            # Pad to make total match remaining_to_assign
            total_deserved = sum(extra_deserved)
            if total_deserved == 0:
                # All tiers had 0 or 1 initial steps; distribute evenly
                extra_deserved = [1] * 4
                total_deserved = 4

            # Normalise to remaining_to_assign
            extra = [int(d * remaining_to_assign / total_deserved) for d in extra_deserved]

            # Distribute rounding errors via rotating dispatch
            deficit = remaining_to_assign - sum(extra)
            assigned = list(extra)
            rotator = DispatchRotator(n_tiers=4)
            for _ in range(deficit):
                order = rotator.current_order()
                for ti in order:
                    if assigned[ti] > 0 or init_counts[ti] > 0:
                        assigned[ti] += 1
                        break
                rotator.advance()

            final_counts = [final_counts[ti] + assigned[ti] for ti in range(4)]

        # Clip to num in case of overshoot
        total = sum(final_counts)
        if total > num:
            # Trim from deficit tier first
            for ti in [3, 2, 1, 0]:
                while total > num and final_counts[ti] > 0:
                    final_counts[ti] -= 1
                    total -= 1
        elif total < num:
            final_counts[3] += num - total

        # --- 6. generate steps per tier region ---
        tier_groups: list[torch.Tensor] = []
        tier_order = [Tier.PRIORITY, Tier.NORMAL, Tier.LOW, Tier.DEFICIT]

        for tier in tier_order:
            ti = tier.value
            count = final_counts[ti]
            if count <= 0:
                continue
            for seg_tier, seg_lo, seg_hi in segments:
                if seg_tier == tier:
                    steps = _karras_in_range(count, seg_lo, seg_hi, self.rho)
                    tier_groups.append(steps)
                    break

        # -- 7. concatenate, sort descending, append 0 ---
        if not tier_groups:
            # fallback — plain Karras + trailing 0 (base already has it)
            self.step_tiers = [3] * num  # all deficit as fallback
            return base

        schedule = torch.cat(tier_groups)
        # Sort ensures monotonic decreasing
        schedule = torch.sort(schedule, descending=True).values
        # Guarantee first value == sigma_max
        schedule[0] = self.sigma_max
        # Guarantee last non-zero value == sigma_min
        if len(schedule) > 1:
            schedule[-1] = self.sigma_min

        # Append the zero endpoint
        schedule = torch.cat([schedule, torch.zeros(1)])

        # Trim to num_steps + 1
        if len(schedule) > num + 1:
            schedule = schedule[:num + 1]
        elif len(schedule) < num + 1:
            # Pad with interpolated values
            pad = num + 1 - len(schedule)
            schedule = torch.cat([schedule, torch.zeros(pad)])

        # --- 8. Compute per-step tier info for solver adaptation ---
        # Budget accumulation uses the SAME BudgetAccumulator params as
        # the schedule itself (budget_max, budget_min, tier_thresholds),
        # ensuring _compute_step_tiers in sampler.py produces identical
        # results when called with default params.
        self.step_tiers: list[int] = []
        final_no_zero = schedule[:-1]
        tier_map = {'priority': 0, 'normal': 1, 'low': 2, 'deficit': 3}
        tier_acc = BudgetAccumulator(
            budget_max=self.budget_max,
            budget_min=self.budget_min,
            tier_thresholds=self.tier_thresholds,
        )
        for i in range(len(final_no_zero) - 1):
            delta = final_no_zero[i] - final_no_zero[i + 1]
            budget = tier_acc.accumulate(delta, final_no_zero[i], self.sigma_max)
            tier_str = tier_acc.classify_tier(budget)
            self.step_tiers.append(tier_map[tier_str])
        # Last step classification: use the accumulated budget
        self.step_tiers.append(tier_map[tier_acc.classify_tier()])

        return schedule
