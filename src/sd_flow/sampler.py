"""
Flow sampler — a fully faithful adaptation of the scx_flow CPU scheduler.

In scx_flow, a task's budget determines its time slice:
  PRIORITY → longest slice (250us)
  NORMAL   → medium slice
  LOW      → short slice
  DEFICIT  → minimum slice (50us)

In sd-flow, a step's budget determines its ODE solver quality:
  PRIORITY → Heun's 2nd order (2 NFEs)
  NORMAL   → Heun's 2nd order (2 NFEs)
  LOW      → Euler (1 NFE)
  DEFICIT  → Fast Euler (1 NFE, no noise injection)

The budget for each step is pre-computed during schedule generation.
Higher-budget steps get more accurate solving (Heun correction);
lower-budget steps run faster (Euler).  This mirrors scx_flow's
variable-slice allocation while keeping the ODE direction correct.
"""

import torch
from tqdm import trange

from .budget import BudgetAccumulator
from .schedule import FlowSigmaSchedule
from .utils import to_d


_TIER_MAP = {'priority': 0, 'normal': 1, 'low': 2, 'deficit': 3}


def _compute_step_tiers(sigmas: torch.Tensor, sigma_max: float) -> list[int]:
    """
    Recompute per-step tier indices from a sigma tensor.

    This is used when the sampler receives a pre-computed sigma tensor
    (e.g. from ComfyUI's KSampler) and needs to determine tier info
    on-the-fly.
    """
    from .budget import BudgetAccumulator
    accumulator = BudgetAccumulator()
    step_tiers: list[int] = []
    for i in range(len(sigmas) - 1):
        delta = sigmas[i] - sigmas[i + 1]
        budget = accumulator.accumulate(delta, sigmas[i], sigma_max)
        tier_str = accumulator.classify_tier(budget)
        step_tiers.append(_TIER_MAP[tier_str])
    return step_tiers


@torch.no_grad()
def sample_flow(model, x, sigmas, extra_args=None, callback=None, disable=None,
                s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """
    Flow ODE sampler — adaptive solver per step, determined by budget tier.

    This is a single unified sampler that replaces both "flow_heun" and
    "flow_euler".  The solver quality for each step is determined by
    the step's flow budget tier:

      Tier        | Solver     | NFEs | Behaviour
      ------------|------------|------|----------
      PRIORITY (0)| Heun       |  2   | Full prediction + correction
      NORMAL   (1)| Heun       |  2   | Full prediction + correction
      LOW      (2)| Euler      |  1   | Prediction only
      DEFICIT  (3)| Fast Euler |  1   | Prediction only, no churn

    Args:
        model: callable(x, sigma, **extra_args) → denoised
        x: noisy latent tensor (B, C, H, W)
        sigmas: 1D sigma tensor [sigma_max, …, 0]
        extra_args: additional kwargs forwarded to model
        callback: optional progress callback
        disable: disable tqdm progress bar
        s_churn: stochastic churn strength (ignored for DEFICIT)
        s_tmin: minimum sigma for churn
        s_tmax: maximum sigma for churn
        s_noise: noise multiplier for churn

    Returns:
        denoised latent tensor
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    # ── Compute per-step tiers ──────────────────────────────────────────
    sigma_max_f = float(sigmas[0])
    step_tiers = _compute_step_tiers(sigmas, sigma_max_f)
    num_steps = len(sigmas) - 1

    for i in trange(num_steps, disable=disable):
        sigma_cur = sigmas[i]
        sigma_next = sigmas[i + 1]
        tier = step_tiers[i]

        # ── Stochastic churn (skipped for DEFICIT) ──────────────────────
        if tier <= 2:  # PRIORITY / NORMAL / LOW
            gamma = min(s_churn / num_steps, 2 ** 0.5 - 1) if s_churn > 0 else 0.0
            if s_tmin <= sigma_cur <= s_tmax and gamma > 0:
                sigma_hat = sigma_cur + gamma * sigma_cur
                eps = torch.randn_like(x) * s_noise
                x = x + (sigma_hat ** 2 - sigma_cur ** 2) ** 0.5 * eps
            else:
                sigma_hat = sigma_cur
        else:  # DEFICIT — no churn, run lean
            sigma_hat = sigma_cur

        # ── 1st order (Euler) prediction (always) ───────────────────────
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d_cur = to_d(x, sigma_hat, denoised)
        dt = sigma_next - sigma_hat

        # ── 2nd order correction only for PRIORITY / NORMAL ─────────────
        if tier <= 1 and i < num_steps - 1:  # PRIORITY or NORMAL
            x_pred = x + dt * d_cur
            denoised_next = model(x_pred, sigma_next * s_in, **extra_args)
            d_prime = to_d(x_pred, sigma_next, denoised_next)
            x = x + dt * (0.5 * d_cur + 0.5 * d_prime)
        else:  # LOW or DEFICIT — Euler only
            x = x + dt * d_cur

        # ── Callback ─────────────────────────────────────────────────────
        if callback is not None:
            callback({
                'x': x,
                'i': i,
                'sigma': sigma_cur,
                'sigma_hat': sigma_hat,
                'denoised': denoised,
            })

    return x


@torch.no_grad()
def sample_flow_heun(model, x, sigmas, extra_args=None, callback=None, disable=None,
                     s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """
    Flow-based ODE sampler using Heun's 2nd order method for every step.

    This is kept for backward compatibility.  New users should prefer
    ``sample_flow`` which adapts the solver per tier.
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_cur = sigmas[i]
        sigma_next = sigmas[i + 1]

        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1)
        if s_tmin <= sigma_cur <= s_tmax:
            sigma_hat = sigma_cur + gamma * sigma_cur
        else:
            sigma_hat = sigma_cur
            gamma = 0.0

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + (sigma_hat ** 2 - sigma_cur ** 2) ** 0.5 * eps

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
            callback({
                'x': x,
                'i': i,
                'sigma': sigma_cur,
                'sigma_hat': sigma_hat,
                'denoised': denoised,
            })

    return x


@torch.no_grad()
def sample_flow_euler(model, x, sigmas, extra_args=None, callback=None, disable=None,
                      s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """
    Flow-based ODE sampler using Euler's method for every step.

    Kept for backward compatibility.  New users should prefer
    ``sample_flow`` which adapts the solver per tier.
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_cur = sigmas[i]
        sigma_next = sigmas[i + 1]

        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1)
        if s_tmin <= sigma_cur <= s_tmax:
            sigma_hat = sigma_cur + gamma * sigma_cur
        else:
            sigma_hat = sigma_cur
            gamma = 0.0

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + (sigma_hat ** 2 - sigma_cur ** 2) ** 0.5 * eps

        denoised = model(x, sigma_hat * s_in, **extra_args)
        d_cur = to_d(x, sigma_hat, denoised)
        x = x + (sigma_next - sigma_hat) * d_cur

        if callback is not None:
            callback({
                'x': x,
                'i': i,
                'sigma': sigma_cur,
                'sigma_hat': sigma_hat,
                'denoised': denoised,
            })

    return x


SAMPLER_FN_MAP = {
    'default': sample_flow,
    'flow': sample_flow,
    'heun': sample_flow_heun,
    'euler': sample_flow_euler,
}


class FlowSampler:
    """
    ODE sampler using the flow scheduling algorithm.

    Compatible with the k-diffusion model call pattern:
        denoised = model(noisy_latents, sigma * ones, **extra_args)

    By default uses the adaptive ``flow`` solver (tier-aware per-step
    solver selection).  Fall back to ``heun`` or ``euler`` for the
    non-adaptive variants.
    """

    def __init__(
        self,
        schedule: FlowSigmaSchedule = None,
        solver: str = 'flow',
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float('inf'),
        s_noise: float = 1.0,
    ):
        self.schedule = schedule or FlowSigmaSchedule()
        self.solver = solver
        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sample(self, denoiser_fn, latents, num_steps=None, sigma_min=None, sigma_max=None):
        """
        Run the flow-based ODE solver.

        Args:
            denoiser_fn: callable(x, sigma) → denoised
            latents: noisy latent tensor (B, C, H, W)
            num_steps: override num_steps
            sigma_min: override sigma_min
            sigma_max: override sigma_max

        Returns:
            denoised latent tensor
        """
        schedule = self.schedule
        if num_steps is not None or sigma_min is not None or sigma_max is not None:
            schedule = FlowSigmaSchedule(
                num_steps or schedule.num_steps,
                sigma_min or schedule.sigma_min,
                sigma_max or schedule.sigma_max,
                schedule.rho,
                schedule.budget_max,
                schedule.budget_min,
                schedule.tier_thresholds,
            )

        sigmas = schedule.generate_schedule()
        sampler_fn = SAMPLER_FN_MAP.get(self.solver, sample_flow)
        return sampler_fn(
            denoiser_fn, latents, sigmas,
            s_churn=self.s_churn,
            s_tmin=self.s_tmin,
            s_tmax=self.s_tmax,
            s_noise=self.s_noise,
        )
