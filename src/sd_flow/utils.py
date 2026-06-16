import torch


def clamp(value, lo, hi):
    """Clamp value to [lo, hi]. Works with scalars or tensors."""
    if isinstance(value, torch.Tensor):
        return torch.clamp(value, lo, hi)
    return max(lo, min(hi, value))


def to_d(x, sigma, denoised):
    """Convert denoiser output to the ODE derivative: (x - denoised) / sigma."""
    return (x - denoised) / sigma


def round_sigma(sigma, decimals=4):
    """Round sigma values for display/comparison."""
    return round(float(sigma), decimals)
