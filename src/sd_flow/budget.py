from .utils import clamp


class BudgetAccumulator:
    """
    Tracks noise budget across sigma transitions.

    Analogous to scx_flow's per-task budget: budget accumulates as sigma
    decreases (the "sleep" phase) and is "spent" during ODE steps.

    The budget reflects how much "noise headroom" remains for a given
    diffusion trajectory. Larger sigma transitions refill more budget;
    smaller transitions refill less.
    """

    def __init__(self, budget_max=2.0, budget_min=-0.5, tier_thresholds=(1.5, 1.0, 0.5)):
        self.budget = 0.0
        self.budget_max = budget_max
        self.budget_min = budget_min
        self.tier_thresholds = tier_thresholds  # (priority, normal, low)

    _REFILL_SCALE = 4.0  # scales refill so cumulative budget spans [0, BUDGET_MAX]

    def refill(self, delta_sigma, sigma_cur, sigma_max, weight=1.0):
        """
        Calculate budget refill from a sigma transition.

        Formula adapted from scx_flow:
            refill = (delta_sigma / sigma_max) * (sigma_cur / sigma_max)
                     * _REFILL_SCALE * weight

        Large drops at high sigma levels accumulate the most budget,
        matching the intuition that early denoising steps have the most
        "creative freedom".

        Args:
            delta_sigma: change in sigma during this transition (always
                         positive as sigma decreases)
            sigma_cur: current sigma value before the transition
            sigma_max: maximum sigma for normalization
            weight: scaling factor for the refill amount

        Returns:
            refill amount (positive = budget increase)
        """
        refill = (
            (delta_sigma / sigma_max)
            * (sigma_cur / sigma_max)
            * self._REFILL_SCALE
            * weight
        )
        return refill

    def accumulate(self, delta_sigma, sigma_cur, sigma_max, weight=1.0):
        """
        Accumulate budget for one sigma transition.

        Computes the refill, adds it to the running budget, and clamps
        the result to [budget_min, budget_max].

        Args:
            delta_sigma: change in sigma during this transition
            sigma_cur: current sigma value before the transition
            sigma_max: maximum sigma for normalization
            weight: scaling factor for the refill amount

        Returns:
            the new budget value after accumulation
        """
        refill = self.refill(delta_sigma, sigma_cur, sigma_max, weight)
        self.budget = clamp(self.budget + refill, self.budget_min, self.budget_max)
        return self.budget

    def classify_tier(self, budget=None):
        """
        Classify the current budget into a tier.

        Returns one of: 'priority', 'normal', 'low', 'deficit'

        Args:
            budget: budget value to classify (defaults to self.budget)
        """
        if budget is None:
            budget = self.budget
        p_thresh, n_thresh, l_thresh = self.tier_thresholds
        if budget >= p_thresh:
            return 'priority'
        elif budget >= n_thresh:
            return 'normal'
        elif budget >= l_thresh:
            return 'low'
        else:
            return 'deficit'

    def reset(self):
        """Reset the budget to zero."""
        self.budget = 0.0
