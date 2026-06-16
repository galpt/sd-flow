"""Tests for sampler functionality."""

import pytest
import torch
from sd_flow.sampler import (
    sample_flow,
    sample_flow_heun,
    sample_flow_euler,
    FlowSampler,
    SAMPLER_FN_MAP,
)
from sd_flow.schedule import FlowSigmaSchedule


# ---------------------------------------------------------------------------
# Function signatures
# ---------------------------------------------------------------------------

class TestSampleFlowHeunSignature:
    """sample_flow_heun function signature."""

    def test_is_callable(self):
        assert callable(sample_flow_heun)

    def test_has_kdiffusion_params(self):
        """Verify signature matches k-diffusion convention."""
        import inspect
        sig = inspect.signature(sample_flow_heun)
        params = list(sig.parameters.keys())
        expected_start = ['model', 'x', 'sigmas']
        for i, p in enumerate(expected_start):
            assert params[i] == p, (
                f"Expected param {i} to be '{p}', got '{params[i]}'"
            )

    def test_has_extra_args(self):
        import inspect
        sig = inspect.signature(sample_flow_heun)
        assert 'extra_args' in sig.parameters

    def test_has_callback_and_disable(self):
        import inspect
        sig = inspect.signature(sample_flow_heun)
        assert 'callback' in sig.parameters
        assert 'disable' in sig.parameters

    def test_has_s_churn_params(self):
        import inspect
        sig = inspect.signature(sample_flow_heun)
        assert 's_churn' in sig.parameters
        assert 's_tmin' in sig.parameters
        assert 's_tmax' in sig.parameters
        assert 's_noise' in sig.parameters

    def test_default_s_churn_zero(self):
        import inspect
        sig = inspect.signature(sample_flow_heun)
        assert sig.parameters['s_churn'].default == 0.0

    def test_default_s_noise_one(self):
        import inspect
        sig = inspect.signature(sample_flow_heun)
        assert sig.parameters['s_noise'].default == 1.0

    @torch.no_grad()
    def test_returns_tensor(self):
        def dummy_denoiser(x, sigma, **kwargs):
            return x * 0.5
        sigmas = torch.tensor([80.0, 40.0, 0.0], dtype=torch.float64)
        x = torch.randn(1, 4, 8, 8) * 80.0
        result = sample_flow_heun(dummy_denoiser, x, sigmas, disable=True)
        assert isinstance(result, torch.Tensor)

    @torch.no_grad()
    def test_output_same_shape_as_input(self):
        def dummy_denoiser(x, sigma, **kwargs):
            return x * 0.5
        sigmas = torch.tensor([80.0, 40.0, 0.0], dtype=torch.float64)
        x = torch.randn(1, 4, 8, 8) * 80.0
        result = sample_flow_heun(dummy_denoiser, x, sigmas, disable=True)
        assert result.shape == x.shape


class TestSampleFlowEulerSignature:
    """sample_flow_euler function signature."""

    def test_is_callable(self):
        assert callable(sample_flow_euler)

    def test_has_kdiffusion_params(self):
        import inspect
        sig = inspect.signature(sample_flow_euler)
        params = list(sig.parameters.keys())
        expected_start = ['model', 'x', 'sigmas']
        for i, p in enumerate(expected_start):
            assert params[i] == p

    def test_has_extra_args(self):
        import inspect
        sig = inspect.signature(sample_flow_euler)
        assert 'extra_args' in sig.parameters

    def test_has_callback_and_disable(self):
        import inspect
        sig = inspect.signature(sample_flow_euler)
        assert 'callback' in sig.parameters
        assert 'disable' in sig.parameters

    def test_has_s_churn_params(self):
        import inspect
        sig = inspect.signature(sample_flow_euler)
        assert 's_churn' in sig.parameters
        assert 's_tmin' in sig.parameters
        assert 's_tmax' in sig.parameters
        assert 's_noise' in sig.parameters

    @torch.no_grad()
    def test_returns_tensor(self):
        def dummy_denoiser(x, sigma, **kwargs):
            return x * 0.5
        sigmas = torch.tensor([80.0, 40.0, 0.0], dtype=torch.float64)
        x = torch.randn(1, 4, 8, 8) * 80.0
        result = sample_flow_euler(dummy_denoiser, x, sigmas, disable=True)
        assert isinstance(result, torch.Tensor)

    @torch.no_grad()
    def test_output_same_shape_as_input(self):
        def dummy_denoiser(x, sigma, **kwargs):
            return x * 0.5
        sigmas = torch.tensor([80.0, 40.0, 0.0], dtype=torch.float64)
        x = torch.randn(1, 4, 8, 8) * 80.0
        result = sample_flow_euler(dummy_denoiser, x, sigmas, disable=True)
        assert result.shape == x.shape


# ---------------------------------------------------------------------------
# SAMPLER_FN_MAP
# ---------------------------------------------------------------------------

class TestSamplerFnMap:
    """SAMPLER_FN_MAP dictionary."""

    def test_contains_heun(self):
        assert 'heun' in SAMPLER_FN_MAP
        assert SAMPLER_FN_MAP['heun'] is sample_flow_heun

    def test_contains_euler(self):
        assert 'euler' in SAMPLER_FN_MAP
        assert SAMPLER_FN_MAP['euler'] is sample_flow_euler

    def test_contains_flow(self):
        assert 'flow' in SAMPLER_FN_MAP
        assert SAMPLER_FN_MAP['flow'] is sample_flow

    def test_has_three_entries(self):
        assert len(SAMPLER_FN_MAP) == 3

    def test_values_are_callable(self):
        for name, fn in SAMPLER_FN_MAP.items():
            assert callable(fn), f"{name} is not callable"


# ---------------------------------------------------------------------------
# FlowSampler construction
# ---------------------------------------------------------------------------

class TestFlowSamplerInit:
    """FlowSampler construction."""

    def test_default_construction(self):
        sampler = FlowSampler()
        assert isinstance(sampler.schedule, FlowSigmaSchedule)
        assert sampler.solver == 'flow'
        assert sampler.s_churn == 0.0
        assert sampler.s_tmin == 0.0
        assert sampler.s_tmax == float('inf')
        assert sampler.s_noise == 1.0

    def test_custom_schedule(self):
        schedule = FlowSigmaSchedule(num_steps=10)
        sampler = FlowSampler(schedule=schedule)
        assert sampler.schedule is schedule

    def test_solver_euler(self):
        sampler = FlowSampler(solver='euler')
        assert sampler.solver == 'euler'

    def test_s_churn_nonzero(self):
        sampler = FlowSampler(s_churn=0.5)
        assert sampler.s_churn == 0.5

    def test_s_tmin(self):
        sampler = FlowSampler(s_tmin=0.1)
        assert sampler.s_tmin == 0.1

    def test_s_tmax(self):
        sampler = FlowSampler(s_tmax=10.0)
        assert sampler.s_tmax == 10.0

    def test_s_noise(self):
        sampler = FlowSampler(s_noise=0.8)
        assert sampler.s_noise == 0.8

    def test_all_params(self):
        schedule = FlowSigmaSchedule(num_steps=10)
        sampler = FlowSampler(
            schedule=schedule,
            solver='euler',
            s_churn=0.2,
            s_tmin=0.05,
            s_tmax=50.0,
            s_noise=1.1,
        )
        assert sampler.schedule is schedule
        assert sampler.solver == 'euler'
        assert sampler.s_churn == 0.2
        assert sampler.s_tmin == 0.05
        assert sampler.s_tmax == 50.0
        assert sampler.s_noise == 1.1


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestFlowSamplerIntegration:
    """End-to-end tests with dummy denoiser."""

    def create_dummy_denoiser(self):
        def dummy_denoiser(x, sigma, **kwargs):
            return x * 0.5
        return dummy_denoiser

    @torch.no_grad()
    def test_sample_euler_basic(self):
        """Integration test: sample_flow_euler with dummy denoiser and
        basic random input returns correct shape."""
        def dummy_denoiser(x, sigma, **kwargs):
            return x * 0.5

        sigmas = torch.tensor([80.0, 40.0, 10.0, 1.0, 0.0], dtype=torch.float64)
        x = torch.randn(1, 4, 8, 8) * 80.0
        result = sample_flow_euler(dummy_denoiser, x, sigmas, disable=True)
        assert result.shape == x.shape
        assert isinstance(result, torch.Tensor)

    @torch.no_grad()
    def test_sample_heun_basic(self):
        """Integration test: sample_flow_heun with dummy denoiser."""
        def dummy_denoiser(x, sigma, **kwargs):
            return x * 0.5

        sigmas = torch.tensor([80.0, 40.0, 10.0, 1.0, 0.0], dtype=torch.float64)
        x = torch.randn(1, 4, 8, 8) * 80.0
        result = sample_flow_heun(dummy_denoiser, x, sigmas, disable=True)
        assert result.shape == x.shape

    @torch.no_grad()
    def test_sample_euler_float32_input(self):
        """Sampler should work with float32 input tensors."""
        def dummy_denoiser(x, sigma, **kwargs):
            return x * 0.5
        sigmas = torch.tensor([80.0, 40.0, 0.0], dtype=torch.float64)
        x = torch.randn(1, 4, 8, 8, dtype=torch.float32) * 80.0
        result = sample_flow_euler(dummy_denoiser, x, sigmas, disable=True)
        assert result.shape == x.shape
        assert result.dtype == torch.float32

    @torch.no_grad()
    def test_sample_heun_float32_input(self):
        """Heun sampler should work with float32 input."""
        def dummy_denoiser(x, sigma, **kwargs):
            return x * 0.5
        sigmas = torch.tensor([80.0, 40.0, 0.0], dtype=torch.float64)
        x = torch.randn(1, 4, 8, 8, dtype=torch.float32) * 80.0
        result = sample_flow_heun(dummy_denoiser, x, sigmas, disable=True)
        assert result.shape == x.shape

    @torch.no_grad()
    def test_flow_sampler_sample_euler(self):
        """FlowSampler.sample() with euler solver."""
        def dummy_denoiser(x, sigma, **kwargs):
            return x * 0.5
        sampler = FlowSampler(solver='euler')
        x = torch.randn(1, 4, 8, 8) * 80.0
        result = sampler.sample(dummy_denoiser, x, num_steps=5)
        assert result.shape == x.shape

    @torch.no_grad()
    def test_flow_sampler_sample_heun(self):
        """FlowSampler.sample() with default adaptive flow solver."""
        def dummy_denoiser(x, sigma, **kwargs):
            return x * 0.5
        sampler = FlowSampler()
        x = torch.randn(1, 4, 8, 8) * 80.0
        result = sampler.sample(dummy_denoiser, x, num_steps=5)
        assert result.shape == x.shape

    @torch.no_grad()
    def test_flow_sampler_custom_sigma_range(self):
        """FlowSampler.sample() with custom sigma range."""
        def dummy_denoiser(x, sigma, **kwargs):
            return x * 0.5
        sampler = FlowSampler(solver='euler')
        x = torch.randn(1, 4, 8, 8) * 50.0
        result = sampler.sample(
            dummy_denoiser, x, num_steps=5,
            sigma_min=0.01, sigma_max=50.0
        )
        assert result.shape == x.shape

    @torch.no_grad()
    def test_result_is_finite(self):
        """Output of sampler should not contain NaN or Inf."""
        def dummy_denoiser(x, sigma, **kwargs):
            return x * 0.5
        sigmas = torch.tensor([80.0, 40.0, 0.0], dtype=torch.float64)
        x = torch.randn(1, 4, 8, 8) * 80.0
        result = sample_flow_euler(dummy_denoiser, x, sigmas, disable=True)
        assert torch.isfinite(result).all()

    @torch.no_grad()
    def test_result_not_all_same(self):
        """Output should not be constant (denoiser reduces noise a bit)."""
        def dummy_denoiser(x, sigma, **kwargs):
            return x * 0.5
        sigmas = torch.tensor([80.0, 40.0, 10.0, 1.0, 0.0], dtype=torch.float64)
        x = torch.randn(1, 4, 8, 8) * 80.0
        result = sample_flow_euler(dummy_denoiser, x, sigmas, disable=True)
        # Not all values should be identical
        assert not (result == result[0, 0, 0, 0]).all()

    @torch.no_grad()
    def test_disable_tqdm(self):
        """disable=True should suppress tqdm progress bar."""
        def dummy_denoiser(x, sigma, **kwargs):
            return x * 0.5
        sigmas = torch.tensor([80.0, 40.0, 0.0], dtype=torch.float64)
        x = torch.randn(1, 4, 8, 8) * 80.0
        # Should run without error
        result = sample_flow_euler(dummy_denoiser, x, sigmas, disable=True)
        assert result.shape == x.shape

    @torch.no_grad()
    def test_extra_args_passed(self):
        """extra_args should be passed through to the denoiser."""
        received = {}

        def capturing_denoiser(x, sigma, **kwargs):
            received.update(kwargs)
            return x * 0.5

        sigmas = torch.tensor([80.0, 40.0, 0.0], dtype=torch.float64)
        x = torch.randn(1, 4, 8, 8) * 80.0
        sample_flow_euler(
            capturing_denoiser, x, sigmas,
            extra_args={'guidance': 3.0, 'prompt': 'test'},
            disable=True,
        )
        assert 'guidance' in received
        assert received['guidance'] == 3.0
        assert 'prompt' in received
        assert received['prompt'] == 'test'
