from enum import Enum


class Tier(Enum):
    PRIORITY = 0
    NORMAL = 1
    LOW = 2
    DEFICIT = 3


# Thresholds for sigma range segmentation (fraction of sigma_max)
# These define the boundaries between tiers in sigma space
TIER_SIGMA_FRACTIONS = {
    Tier.PRIORITY: (0.75, 1.0),   # top 25%: highest noise
    Tier.NORMAL:   (0.50, 0.75),  # next 25%
    Tier.LOW:      (0.25, 0.50),  # next 25%
    Tier.DEFICIT:  (0.0,  0.25),  # bottom 25%: lowest noise
}


def segment_sigma_range(sigma_min, sigma_max):
    """
    Divide [sigma_min, sigma_max] into 4 tier regions.

    Returns: list of (Tier, sigma_lo, sigma_hi) sorted descending by sigma.
    The ranges cover [sigma_min, sigma_max] without gaps.

    Each tier's sigma range is derived from its fraction of sigma_max,
    with sigma_min acting as the floor for the DEFICIT tier to ensure
    full coverage of the input range.
    """
    regions = []
    # Iterate in descending sigma order
    tiers_in_order = [Tier.PRIORITY, Tier.NORMAL, Tier.LOW, Tier.DEFICIT]
    for tier in tiers_in_order:
        frac_lo, frac_hi = TIER_SIGMA_FRACTIONS[tier]
        # Convert fraction boundaries to sigma space
        sigma_hi = frac_hi * sigma_max
        sigma_lo = max(frac_lo * sigma_max, sigma_min)
        regions.append((tier, sigma_lo, sigma_hi))
    return regions
