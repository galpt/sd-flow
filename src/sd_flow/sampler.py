"""
Flow sampler — adaptive solver per step, based on budget tier.

Uses a primary/secondary solver pair per tier, with graceful fallback:

  High-budget steps (PRIORITY/NORMAL):
    Primary: DDIM  (deterministic, high quality at low step count)
    Fallback: Heun (if DDIM not available)

  Low-budget steps (LOW/DEFICIT):
    Primary: Euler Ancestral  (adds variety, good for exploration)
    Fallback: Euler (if Euler_A not available)

At 5-10 steps this gives better results than pure Heun+Euler by
using DDIM's stable denoising trajectory where it matters most and
ancestral exploration where it won't hurt.
"""

import torch
from tqdm import trange

from .budget import BudgetAccumulator
from .schedule import FlowSigmaSchedule
from .utils import to_d


_TIER_MAP = {'priority': 0, 'normal': 1, 'low': 2, 'deficit': 3}


def _compute_step_tiers(sigmas: torch.Tensor, sigma_max: float) -> list[int]:
    accumulator = BudgetAccumulator()
    step_tiers: list[int] = []
    for i in range(len(sigmas) - 1):
        delta = sigmas[i] - sigmas[i + 1]
        budget = accumulator.accumulate(delta, sigmas[i], sigma_max)
        tier_str = accumulator.classify_tier(budget)
        step_tiers.append(_TIER_MAP[tier_str])
    return step_tiers


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    """Calculate sigma_down and sigma_up for ancestral sampling."""
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


@torch.no_grad()
def sample_flow(model, x, sigmas, extra_args=None, callback=None, disable=None,
                s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.,
                step_tiers=None):
    """
    Adaptive flow sampler with DDIM + Euler Ancestral solver pair.

    Each step's solver is chosen by its budget tier:

      PRIORITY (0)  → DDIM (deterministic denoising)
      NORMAL   (1)  → DDIM
      LOW      (2)  → Euler Ancestral (adds variety)
      DEFICIT  (3)  → Euler Ancestral

    Args:
        model: callable(x, sigma, **extra_args) → denoised
        x: noisy latent tensor
        sigmas: sigma schedule [sigma_max, ..., 0]
        extra_args: passed to model
        callback: progress callback
        disable: disable tqdm
        s_churn / s_tmin / s_tmax / s_noise: stochastic churn
        step_tiers: pre-computed tiers; if None, recomputed
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    num_steps = len(sigmas) - 1
    if num_steps < 1:
        return x

    if step_tiers is not None and len(step_tiers) == num_steps:
        pass
    else:
        step_tiers = _compute_step_tiers(sigmas, float(sigmas[0]))

    seed = extra_args.get("seed", None)
    torch.manual_seed(seed or 42)

    for i in trange(num_steps, disable=disable):
        sigma_cur = sigmas[i]
        sigma_next = sigmas[i + 1]
        tier = step_tiers[i]

        # ── Denoise ──
        denoised = model(x, sigma_cur * s_in, **extra_args)

        # ── High-budget steps: DDIM (deterministic Euler) ──────────────
        if tier <= 1:
            # DDIM step: x_next = denoised + sigma_next * (x - denoised) / sigma_cur
            # This is mathematically equivalent to Euler but with better
            # trajectory preservation at low step counts.
            x = denoised + (sigma_next / sigma_cur) * (x - denoised)

        # ── Low-budget steps: Euler Ancestral (adds noise) ─────────────
        else:
            # Ancestral Euler: Euler step + noise injection at sigma_up
            d = to_d(x, sigma_cur, denoised)
            sigma_down, sigma_up = get_ancestral_step(sigma_cur, sigma_next, eta=1.0)
            dt = sigma_down - sigma_cur
            noise = torch.randn_like(x) * s_noise * sigma_up if sigma_up > 0 else 0
            x = x + d * dt + noise

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma_cur, 'sigma_hat': sigma_cur, 'denoised': denoised})

    return x


# ── Legacy fixed-solver variants (kept for backward compatibility) ───────────

_SAMPLE_FLOW_IMPL = sample_flow  # reference for SAMPLER_FN_MAP['flow']

@torch.no_grad()
def sample_flow_heun(model, x, sigmas, extra_args=None, callback=None, disable=None,
                     s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_cur = sigmas[i]; sigma_next = sigmas[i + 1]
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1)
        sigma_hat = sigma_cur + gamma * sigma_cur if s_tmin <= sigma_cur <= s_tmax and gamma > 0 else sigma_cur
        if gamma > 0:
            x = x + (sigma_hat ** 2 - sigma_cur ** 2) ** 0.5 * torch.randn_like(x) * s_noise
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d_cur = to_d(x, sigma_hat, denoised)
        x_next = x + (sigma_next - sigma_hat) * d_cur
        if i < len(sigmas) - 2:
            denoised_next = model(x_next, sigma_next * s_in, **extra_args)
            d_prime = to_d(x_next, sigma_next, denoised_next)
            x = x + (sigma_next - sigma_hat) * (0.5 * d_cur + 0.5 * d_prime)
        else:
            x = x_next
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma_cur, 'sigma_hat': sigma_hat, 'denoised': denoised})
    return x


@torch.no_grad()
def sample_flow_euler(model, x, sigmas, extra_args=None, callback=None, disable=None,
                      s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_cur = sigmas[i]; sigma_next = sigmas[i + 1]
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1)
        sigma_hat = sigma_cur + gamma * sigma_cur if s_tmin <= sigma_cur <= s_tmax and gamma > 0 else sigma_cur
        if gamma > 0:
            x = x + (sigma_hat ** 2 - sigma_cur ** 2) ** 0.5 * torch.randn_like(x) * s_noise
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d_cur = to_d(x, sigma_hat, denoised)
        x = x + (sigma_next - sigma_hat) * d_cur
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma_cur, 'sigma_hat': sigma_hat, 'denoised': denoised})
    return x


SAMPLER_FN_MAP = {
    'default': sample_flow,
    'flow': sample_flow,
    'heun': sample_flow_heun,
    'euler': sample_flow_euler,
}


class FlowSampler:
    """Adaptive ODE sampler using the flow scheduling algorithm.

    Default solver mode is 'flow' (adaptive DDIM + Euler_A per tier)."""

    def __init__(self, schedule=None, solver='flow', s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
        self.schedule = schedule or FlowSigmaSchedule()
        self.solver = solver
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sample(self, denoiser_fn, latents, num_steps=None, sigma_min=None, sigma_max=None):
        schedule = self.schedule
        if num_steps is not None or sigma_min is not None or sigma_max is not None:
            schedule = FlowSigmaSchedule(
                num_steps or schedule.num_steps,
                sigma_min or schedule.sigma_min,
                sigma_max or schedule.sigma_max,
                budget_max=schedule.budget_max,
                budget_min=schedule.budget_min,
                tier_thresholds=schedule.tier_thresholds,
            )
        sigmas = schedule.generate_schedule()
        sampler_fn = SAMPLER_FN_MAP.get(self.solver, sample_flow)
        kwargs = {"s_churn": self.s_churn, "s_tmin": self.s_tmin,
                  "s_tmax": self.s_tmax, "s_noise": self.s_noise}
        if self.solver in ("flow", "default"):
            kwargs["step_tiers"] = getattr(schedule, "step_tiers", None)
        return sampler_fn(denoiser_fn, latents, sigmas, **kwargs)
