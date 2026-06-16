import torch
from tqdm import trange

from .schedule import FlowSigmaSchedule
from .utils import to_d


@torch.no_grad()
def sample_flow_heun(model, x, sigmas, extra_args=None, callback=None, disable=None,
                     s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """
    Flow-based ODE sampler using Heun's 2nd order method.

    Matches the k-diffusion sampler function signature so it can be wrapped
    by ComfyUI's KSAMPLER.

    Args:
        model: callable(x, sigma, **extra_args) -> denoised
        x: noisy latent tensor (B, C, H, W)
        sigmas: 1D tensor of sigma values [sigma_max, ..., 0]
        extra_args: dict passed to model
        callback: optional progress callback
        disable: disable tqdm progress bar
        s_churn: stochasticity parameter
        s_tmin: minimum sigma for stochastic churn
        s_tmax: maximum sigma for stochastic churn
        s_noise: noise scale factor

    Returns:
        denoised latent tensor
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_cur = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Stochastic churn (from Karras et al.)
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1)
        if s_tmin <= sigma_cur <= s_tmax:
            sigma_hat = sigma_cur + gamma * sigma_cur
        else:
            sigma_hat = sigma_cur
            gamma = 0.0

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + (sigma_hat ** 2 - sigma_cur ** 2) ** 0.5 * eps

        # 1st order (Euler) prediction
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d_cur = to_d(x, sigma_hat, denoised)
        x_next = x + (sigma_next - sigma_hat) * d_cur

        # 2nd order (Heun) correction
        if i < len(sigmas) - 2:  # skip on last step
            denoised_next = model(x_next, sigma_next * s_in, **extra_args)
            d_prime = to_d(x_next, sigma_next, denoised_next)
            x_next = x + (sigma_next - sigma_hat) * (0.5 * d_cur + 0.5 * d_prime)

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
    Flow-based ODE sampler using Euler's method (1st order).

    Matches the k-diffusion sampler function signature so it can be wrapped
    by ComfyUI's KSAMPLER.

    Args:
        model: callable(x, sigma, **extra_args) -> denoised
        x: noisy latent tensor (B, C, H, W)
        sigmas: 1D tensor of sigma values [sigma_max, ..., 0]
        extra_args: dict passed to model
        callback: optional progress callback
        disable: disable tqdm progress bar
        s_churn: stochasticity parameter
        s_tmin: minimum sigma for stochastic churn
        s_tmax: maximum sigma for stochastic churn
        s_noise: noise scale factor

    Returns:
        denoised latent tensor
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
    'heun': sample_flow_heun,
    'euler': sample_flow_euler,
}


class FlowSampler:
    """
    ODE sampler using the flow sigma schedule.

    Compatible with the k-diffusion model call pattern:
        denoised = model(noisy_latents, sigma * ones, **extra_args)
    """

    def __init__(
        self,
        schedule: FlowSigmaSchedule = None,
        solver: str = 'heun',
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
            denoiser_fn: callable(x, sigma) -> denoised
            latents: noisy latent tensor (B, C, H, W)
            num_steps: override num_steps (optional)
            sigma_min: override sigma_min (optional)
            sigma_max: override sigma_max (optional)

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

        sampler_fn = SAMPLER_FN_MAP.get(self.solver, sample_flow_heun)
        return sampler_fn(
            denoiser_fn, latents, sigmas,
            s_churn=self.s_churn,
            s_tmin=self.s_tmin,
            s_tmax=self.s_tmax,
            s_noise=self.s_noise,
        )
