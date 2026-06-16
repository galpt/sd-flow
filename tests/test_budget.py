"""Tests for BudgetAccumulator."""

import pytest
from sd_flow.budget import BudgetAccumulator


class TestBudgetAccumulatorInit:
    """BudgetAccumulator construction and initial state."""

    def test_initial_budget_is_zero(self):
        acc = BudgetAccumulator()
        assert acc.budget == 0.0

    def test_default_attributes(self):
        acc = BudgetAccumulator()
        assert acc.budget_max == 2.0
        assert acc.budget_min == -0.5
        assert acc.tier_thresholds == (1.5, 1.0, 0.5)

    def test_custom_thresholds(self):
        acc = BudgetAccumulator(
            budget_max=5.0, budget_min=-1.0, tier_thresholds=(3.0, 2.0, 1.0)
        )
        assert acc.budget_max == 5.0
        assert acc.budget_min == -1.0
        assert acc.tier_thresholds == (3.0, 2.0, 1.0)

    def test_custom_budget_clamp_range(self):
        acc = BudgetAccumulator(budget_max=10.0, budget_min=-2.0)
        assert acc.budget_max == 10.0
        assert acc.budget_min == -2.0


class TestBudgetAccumulatorRefill:
    """BudgetAccumulator.refill() correctness."""

    def test_refill_formula(self):
        acc = BudgetAccumulator()
        # refill = (delta_sigma / sigma_max) * (sigma_cur / sigma_max) * weight
        # delta_sigma=20.0, sigma_cur=80.0, sigma_max=80.0, weight=1.0
        # (20.0 / 80.0) * (80.0 / 80.0) * 1.0 = 0.25 * 1.0 * 1.0 = 0.25
        r = acc.refill(20.0, 80.0, 80.0, weight=1.0)
        assert r == pytest.approx(0.25)

    def test_refill_weight_half(self):
        acc = BudgetAccumulator()
        # weight=0.5 → 0.25 * 0.5 = 0.125
        r = acc.refill(20.0, 80.0, 80.0, weight=0.5)
        assert r == pytest.approx(0.125)

    def test_refill_weight_double(self):
        acc = BudgetAccumulator()
        # weight=2.0 → 0.25 * 2.0 = 0.5
        r = acc.refill(20.0, 80.0, 80.0, weight=2.0)
        assert r == pytest.approx(0.5)

    def test_refill_zero_delta(self):
        acc = BudgetAccumulator()
        # delta_sigma=0 -> refill=0
        r = acc.refill(0.0, 80.0, 80.0)
        assert r == pytest.approx(0.0)

    def test_refill_small_delta(self):
        acc = BudgetAccumulator()
        # delta_sigma=0.5, sigma_cur=40.0, sigma_max=80.0
        # (0.5 / 80.0) * (40.0 / 80.0) = 0.00625 * 0.5 = 0.003125
        r = acc.refill(0.5, 40.0, 80.0)
        assert r == pytest.approx(0.003125)

    def test_refill_weight_proportional(self):
        """Verify weight scales refill proportionally."""
        acc = BudgetAccumulator()
        base = acc.refill(10.0, 50.0, 80.0, weight=1.0)
        scaled = acc.refill(10.0, 50.0, 80.0, weight=3.0)
        assert scaled == pytest.approx(base * 3.0)


class TestBudgetAccumulatorAccumulate:
    """BudgetAccumulator.accumulate() correctness."""

    def test_accumulate_increases_budget(self):
        acc = BudgetAccumulator()
        assert acc.budget == 0.0
        new_budget = acc.accumulate(20.0, 80.0, 80.0)
        # refill = 0.25, budget = 0 + 0.25 = 0.25
        assert new_budget == pytest.approx(0.25)
        assert acc.budget == pytest.approx(0.25)

    def test_accumulate_multiple(self):
        acc = BudgetAccumulator()
        acc.accumulate(20.0, 80.0, 80.0)  # +0.25
        acc.accumulate(20.0, 60.0, 80.0)  # (20/80)*(60/80)=0.25*0.75=0.1875
        assert acc.budget == pytest.approx(0.25 + 0.1875)

    def test_accumulate_passes_weight(self):
        acc = BudgetAccumulator()
        new_budget = acc.accumulate(20.0, 80.0, 80.0, weight=2.0)
        # refill = 0.5
        assert new_budget == pytest.approx(0.5)
        assert acc.budget == pytest.approx(0.5)


class TestBudgetAccumulatorClamping:
    """Budget clamping to [budget_min, budget_max]."""

    def test_clamp_above_max(self):
        acc = BudgetAccumulator(budget_max=2.0, budget_min=-0.5)
        # Accumulate enough to exceed budget_max
        # refill per call = (80/80)*(80/80)*1=1.0
        acc.accumulate(80.0, 80.0, 80.0)  # +1.0
        acc.accumulate(80.0, 80.0, 80.0)  # +1.0 -> 2.0
        acc.accumulate(80.0, 80.0, 80.0)  # +1.0 should clamp to 2.0
        assert acc.budget == pytest.approx(2.0)
        assert acc.budget <= acc.budget_max

    def test_clamp_below_min(self):
        acc = BudgetAccumulator(budget_max=2.0, budget_min=-0.5)
        # Negative weight to simulate budget decrease
        acc.accumulate(10.0, 80.0, 80.0, weight=-1.0)  # -0.125
        # Should still be -0.125, above budget_min
        assert acc.budget == pytest.approx(-0.125)
        # Hit the floor
        acc.accumulate(50.0, 80.0, 80.0, weight=-1.0)  # -0.625 -> total -0.75, clamp to -0.5
        assert acc.budget == pytest.approx(-0.5)
        assert acc.budget >= acc.budget_min

    def test_custom_clamp_range(self):
        acc = BudgetAccumulator(budget_max=1.0, budget_min=0.0)
        acc.accumulate(80.0, 80.0, 80.0)  # +1.0 → budget=1.0
        acc.accumulate(10.0, 80.0, 80.0)  # +0.125 → clamp(1.125, 0, 1) = 1.0
        assert acc.budget == pytest.approx(1.0)
        # Accumulate with negative weight to drain budget
        # weight=-1.0: refill = -0.125, budget = clamp(1.0 + -0.125, 0, 1) = 0.875
        # Need multiple accumulations to reach budget_min
        acc.accumulate(10.0, 80.0, 80.0, weight=-1.0)  # -0.125 → 0.875
        assert acc.budget == pytest.approx(0.875)
        acc.accumulate(10.0, 80.0, 80.0, weight=-1.0)  # -0.125 → 0.75
        assert acc.budget == pytest.approx(0.75)
        acc.accumulate(10.0, 80.0, 80.0, weight=-1.0)  # -0.125 → 0.625
        assert acc.budget == pytest.approx(0.625)
        # Eventually hit the floor (budget_min=0.0)
        for _ in range(5):
            acc.accumulate(10.0, 80.0, 80.0, weight=-1.0)
        assert acc.budget == pytest.approx(0.0)


class TestBudgetAccumulatorClassifyTier:
    """BudgetAccumulator.classify_tier()."""

    def test_priority_tier(self):
        acc = BudgetAccumulator()
        assert acc.classify_tier(1.5) == 'priority'
        assert acc.classify_tier(2.0) == 'priority'
        assert acc.classify_tier(10.0) == 'priority'

    def test_normal_tier(self):
        acc = BudgetAccumulator()
        assert acc.classify_tier(1.0) == 'normal'
        assert acc.classify_tier(1.25) == 'normal'
        assert acc.classify_tier(1.49) == 'normal'

    def test_low_tier(self):
        acc = BudgetAccumulator()
        assert acc.classify_tier(0.5) == 'low'
        assert acc.classify_tier(0.75) == 'low'
        assert acc.classify_tier(0.99) == 'low'

    def test_deficit_tier(self):
        acc = BudgetAccumulator()
        assert acc.classify_tier(0.0) == 'deficit'
        assert acc.classify_tier(0.25) == 'deficit'
        assert acc.classify_tier(0.49) == 'deficit'
        assert acc.classify_tier(-1.0) == 'deficit'

    def test_classify_uses_self_budget(self):
        acc = BudgetAccumulator()
        assert acc.classify_tier() == 'deficit'  # initial budget = 0.0
        acc.accumulate(80.0, 80.0, 80.0)  # +1.0
        assert acc.classify_tier() == 'normal'

    def test_custom_thresholds_classify(self):
        acc = BudgetAccumulator(tier_thresholds=(4.0, 3.0, 2.0))
        assert acc.classify_tier(4.0) == 'priority'
        assert acc.classify_tier(3.0) == 'normal'
        assert acc.classify_tier(2.0) == 'low'
        assert acc.classify_tier(1.0) == 'deficit'


class TestBudgetAccumulatorReset:
    """BudgetAccumulator.reset()."""

    def test_reset_sets_budget_to_zero(self):
        acc = BudgetAccumulator()
        acc.accumulate(80.0, 80.0, 80.0)  # budget = 1.0
        assert acc.budget == pytest.approx(1.0)
        acc.reset()
        assert acc.budget == 0.0

    def test_reset_from_clamped_value(self):
        acc = BudgetAccumulator(budget_max=2.0, budget_min=-0.5)
        acc.accumulate(80.0, 80.0, 80.0)  # +1.0
        acc.accumulate(80.0, 80.0, 80.0)  # +1.0 -> 2.0
        acc.accumulate(80.0, 80.0, 80.0)  # +1.0 -> clamped at 2.0
        assert acc.budget == pytest.approx(2.0)
        acc.reset()
        assert acc.budget == 0.0

    def test_accumulate_after_reset(self):
        acc = BudgetAccumulator()
        acc.accumulate(40.0, 80.0, 80.0)  # +0.5
        acc.reset()
        acc.accumulate(20.0, 80.0, 80.0)  # +0.25
        assert acc.budget == pytest.approx(0.25)


class TestBudgetAccumulatorEdgeCases:
    """Edge cases for BudgetAccumulator."""

    def test_negative_delta_sigma(self):
        acc = BudgetAccumulator()
        # Negative delta_sigma (sigma increasing — unusual but handle gracefully)
        r = acc.refill(-10.0, 40.0, 80.0)
        assert r == pytest.approx((-10.0 / 80.0) * (40.0 / 80.0))

    def test_zero_sigma_cur(self):
        acc = BudgetAccumulator()
        r = acc.refill(10.0, 0.0, 80.0)
        assert r == pytest.approx(0.0)  # sigma_cur=0 → term=0

    def test_accumulate_no_side_effect_on_refill(self):
        """accumulate calls refill but does not modify its own budget before adding."""
        acc = BudgetAccumulator()
        r = acc.refill(20.0, 80.0, 80.0)
        assert r == pytest.approx(0.25)
        # refill should be stateless — internal budget unchanged
        assert acc.budget == pytest.approx(0.0)
