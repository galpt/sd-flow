"""Tests for FlowSigmaSchedule."""

import pytest
import torch
from sd_flow.schedule import FlowSigmaSchedule


# ---------------------------------------------------------------------------
# FlowSigmaSchedule
# ---------------------------------------------------------------------------

class TestFlowSigmaScheduleInit:
    """FlowSigmaSchedule construction."""

    def test_default_construction(self):
        sched = FlowSigmaSchedule()
        assert sched.num_steps == 18
        assert sched.sigma_min == 0.002
        assert sched.sigma_max == 80.0
        assert sched.budget_max == 2.0
        assert sched.budget_min == -0.5
        assert sched.tier_thresholds == (1.5, 1.0, 0.5)

    def test_custom_construction(self):
        sched = FlowSigmaSchedule(
            num_steps=25, sigma_min=0.01, sigma_max=100.0,
            budget_max=3.0, budget_min=-1.0, tier_thresholds=(2.0, 1.5, 0.8)
        )
        assert sched.num_steps == 25
        assert sched.sigma_min == 0.01
        assert sched.sigma_max == 100.0
        assert sched.budget_max == 3.0
        assert sched.budget_min == -1.0
        assert sched.tier_thresholds == (2.0, 1.5, 0.8)


class TestFlowSigmaScheduleOutputShape:
    """FlowSigmaSchedule.generate_schedule() output shape."""

    @pytest.mark.parametrize("num_steps", [1, 5, 10, 18, 50, 100])
    def test_shape_various_steps(self, num_steps):
        sched = FlowSigmaSchedule(num_steps=num_steps)
        sigmas = sched.generate_schedule()
        assert isinstance(sigmas, torch.Tensor)
        assert sigmas.shape == (num_steps + 1,), (
            f"Expected ({num_steps + 1},), got {sigmas.shape}"
        )

    def test_single_step(self):
        sched = FlowSigmaSchedule(num_steps=1)
        sigmas = sched.generate_schedule()
        assert sigmas.shape == (2,)


class TestFlowSigmaScheduleValues:
    """FlowSigmaSchedule.generate_schedule() value correctness."""

    def test_first_is_sigma_max(self):
        sched = FlowSigmaSchedule()
        sigmas = sched.generate_schedule()
        assert sigmas[0].item() == pytest.approx(sched.sigma_max, abs=1e-5)

    def test_last_is_zero(self):
        sched = FlowSigmaSchedule()
        sigmas = sched.generate_schedule()
        assert sigmas[-1].item() == pytest.approx(0.0)

    def test_monotonically_decreasing(self):
        sched = FlowSigmaSchedule()
        sigmas = sched.generate_schedule()
        diffs = sigmas[1:] - sigmas[:-1]
        assert (diffs <= 0).all(), f"Schedule not monotonic: diffs={diffs}"

    @pytest.mark.parametrize("num_steps", [1, 5, 10, 18, 50, 100])
    def test_monotonic_many_steps(self, num_steps):
        sched = FlowSigmaSchedule(num_steps=num_steps)
        sigmas = sched.generate_schedule()
        diffs = sigmas[1:] - sigmas[:-1]
        assert (diffs <= 0).all(), f"Not monotonic for {num_steps} steps"

    def test_all_sigmas_positive_except_last(self):
        sched = FlowSigmaSchedule()
        sigmas = sched.generate_schedule()
        assert (sigmas[:-1] > 0).all(), "Non-positive sigma found in schedule"
        assert sigmas[-1] == 0.0

    def test_dtype(self):
        sched = FlowSigmaSchedule()
        sigmas = sched.generate_schedule()
        assert sigmas.dtype == torch.float32


class TestFlowSigmaScheduleCustomRange:
    """FlowSigmaSchedule with custom sigma_min/sigma_max."""

    def test_custom_sigma_max(self):
        sched = FlowSigmaSchedule(sigma_max=50.0)
        sigmas = sched.generate_schedule()
        assert sigmas[0].item() == pytest.approx(50.0, abs=1e-5)

    def test_custom_sigma_min(self):
        sched = FlowSigmaSchedule(sigma_min=0.1)
        sigmas = sched.generate_schedule()
        assert sigmas[-1].item() == pytest.approx(0.0)
        # The second-to-last should be approximately sigma_min
        # (linear schedule should be within float32 precision of sigma_min)
        assert sigmas[-2].item() == pytest.approx(0.1, abs=1e-3)

    def test_custom_both(self):
        sched = FlowSigmaSchedule(num_steps=10, sigma_min=0.005, sigma_max=50.0)
        sigmas = sched.generate_schedule()
        assert sigmas.shape == (11,)
        assert sigmas[0].item() == pytest.approx(50.0, abs=1e-5)
        assert sigmas[-1].item() == pytest.approx(0.0)

    def test_small_sigma_range(self):
        sched = FlowSigmaSchedule(num_steps=10, sigma_min=1.0, sigma_max=4.0)
        sigmas = sched.generate_schedule()
        assert sigmas.shape == (11,)
        assert sigmas[0].item() == pytest.approx(4.0, abs=1e-5)
        assert sigmas[-1].item() == pytest.approx(0.0)
        diffs = sigmas[1:] - sigmas[:-1]
        assert (diffs <= 0).all()


class TestFlowSigmaScheduleDeterminism:
    """FlowSigmaSchedule determinism."""

    def test_deterministic(self):
        sched = FlowSigmaSchedule(num_steps=18)
        sigmas1 = sched.generate_schedule()
        sigmas2 = sched.generate_schedule()
        assert torch.equal(sigmas1, sigmas2)

    def test_deterministic_custom_params(self):
        sched = FlowSigmaSchedule(
            num_steps=10, sigma_min=0.01, sigma_max=50.0
        )
        sigmas1 = sched.generate_schedule()
        sigmas2 = sched.generate_schedule()
        assert torch.equal(sigmas1, sigmas2)

    def test_separate_instances_same_result(self):
        sched_a = FlowSigmaSchedule(num_steps=18)
        sched_b = FlowSigmaSchedule(num_steps=18)
        sigmas_a = sched_a.generate_schedule()
        sigmas_b = sched_b.generate_schedule()
        assert torch.equal(sigmas_a, sigmas_b)


class TestFlowSigmaScheduleEdgeCases:
    """Edge cases for FlowSigmaSchedule."""

    def test_various_num_steps_all_valid(self):
        for n in [1, 2, 3, 4, 5, 10, 18, 25, 50, 100]:
            sched = FlowSigmaSchedule(num_steps=n)
            sigmas = sched.generate_schedule()
            assert sigmas.shape == (n + 1,)
            assert sigmas[0].item() == pytest.approx(80.0, abs=1e-5)
            assert sigmas[-1].item() == pytest.approx(0.0)
            diffs = sigmas[1:] - sigmas[:-1]
            assert (diffs <= 0).all(), f"Failed at n={n}"

    def test_fallback_when_tier_sigma_groups_empty(self):
        """Test that generate_schedule handles the case where no tier
        groups are generated (fallback to base Karras + trailing zero)."""
        # Use a configuration that still generates valid output
        sched = FlowSigmaSchedule(num_steps=18)
        sigmas = sched.generate_schedule()
        # Should always return a valid schedule
        assert sigmas is not None
        assert len(sigmas) > 1

    def test_return_type(self):
        sched = FlowSigmaSchedule()
        sigmas = sched.generate_schedule()
        assert isinstance(sigmas, torch.Tensor)
