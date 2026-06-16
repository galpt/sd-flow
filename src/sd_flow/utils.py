import torch


def clamp(value, lo, hi):
    """Clamp value to [lo, hi]. Works with scalars or tensors."""
    if isinstance(value, torch.Tensor):
        return torch.clamp(value, lo, hi)
    return max(lo, min(hi, value))


def to_d(x, sigma, denoised):
    """
    Convert denoiser output to Karras ODE derivative.

    Formula: (x - denoised) / sigma
    This matches the k-diffusion convention.
    """
    return (x - denoised) / sigma  # sigma broadcast handled by PyTorch


def round_sigma(sigma, decimals=4):
    """Round sigma values for display/comparison."""
    return round(float(sigma), decimals)


def interleave_lists(lists):
    """
    Interleave items from multiple lists in round-robin fashion,
    stopping when all lists are exhausted.
    """
    result = []
    lists = [list(lst) for lst in lists]  # copy
    while any(lists):
        for lst in lists:
            if lst:
                result.append(lst.pop(0))
    return result
