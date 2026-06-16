"""Debug version of schedule generation with print statements."""

import torch
from .budget import BudgetAccumulator
from .tiers import Tier, segment_sigma_range
from .rotating_dispatch import DispatchRotator


def generate_flow_schedule(
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80.0,
    rho=7.0,
    budget_max=2.0,
    budget_min=-0.5,
    tier_thresholds=(1.5, 1.0, 0.5),
):
    """Generate flow-based sigma schedule with debug output."""
    
    # 1. Segment
    print("Step 1: Segment sigma range")
    segments = segment_sigma_range(sigma_min, sigma_max)
    print(f"  Segments: {[(t.name, lo, hi) for t, lo, hi in segments]}")
    
    # 2. Base Karras schedule
    print("Step 2: Base Karras schedule")
    ramp = torch.linspace(0, 1, num_steps)
    min_inv = sigma_min ** (1 / rho)
    max_inv = sigma_max ** (1 / rho)
    base_sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
    print(f"  Base sigmas: {[f'{float(v):.2f}' for v in base_sigmas]}")
    
    # 3. Budget accumulation
    print("Step 3: Budget accumulation")
    accumulator = BudgetAccumulator(
        budget_max=budget_max,
        budget_min=budget_min,
        tier_thresholds=tier_thresholds,
    )
    
    budgets = []
    step_tiers = []
    for i in range(len(base_sigmas) - 1):
        delta = base_sigmas[i] - base_sigmas[i + 1]
        budget = accumulator.accumulate(delta, base_sigmas[i], sigma_max)
        tier = accumulator.classify_tier(budget)
        budgets.append(budget)
        step_tiers.append(tier)
    print(f"  Step tiers: {step_tiers}")
    print(f"  Budgets: {[f'{b:.4f}' for b in budgets]}")
    
    # 4. Count per tier
    print("Step 4: Count per tier")
    tier_map = {'priority': 0, 'normal': 1, 'low': 2, 'deficit': 3}
    tier_idx = [tier_map[t] for t in step_tiers]
    initial_counts = [tier_idx.count(t) for t in range(4)]
    print(f"  Initial counts: {initial_counts}")
    
    # 5. Rotating dispatch
    print("Step 5: Rotating dispatch")
    rotator = DispatchRotator(n_tiers=4)
    final_counts = [0, 0, 0, 0]
    total_allocated = 0
    remaining = list(initial_counts)
    iterations = 0
    
    while total_allocated < num_steps:
        iterations += 1
        if iterations > 100:
            print(f"  ERROR: deadlock detected at iteration {iterations}")
            print(f"    total_allocated={total_allocated}, remaining={remaining}")
            break
        order = rotator.current_order()
        for ti in order:
            if total_allocated >= num_steps:
                break
            if remaining[ti] > 0:
                final_counts[ti] += 1
                remaining[ti] -= 1
                total_allocated += 1
        rotator.advance()
    
    print(f"  Iterations: {iterations}")
    print(f"  Final counts: {final_counts}")
    print(f"  Sum final: {sum(final_counts)}")
    
    leftover = num_steps - sum(final_counts)
    if leftover > 0:
        print(f"  Leftover: {leftover} -> assigning to deficit")
        final_counts[3] += leftover
    
    # 6. Generate steps per tier
    print("Step 6: Generate steps per tier")
    tier_sigmas = []
    tier_order = [Tier.PRIORITY, Tier.NORMAL, Tier.LOW, Tier.DEFICIT]
    
    for tier in tier_order:
        ti = tier.value
        count = final_counts[ti]
        print(f"  Tier {tier.name}: count={count}")
        if count <= 0:
            continue
        for seg_tier, seg_lo, seg_hi in segments:
            if seg_tier == tier:
                steps = _karras_steps_in_range(count, seg_lo, seg_hi, rho)
                print(f"    Range [{seg_lo:.4f}, {seg_hi:.4f}]: {[f'{float(v):.4f}' for v in (steps[:5] if len(steps) > 5 else steps)]}")
                tier_sigmas.append(steps)
                break
    
    if not tier_sigmas:
        print("  No tier sigmas generated, falling back")
        return torch.cat([base_sigmas, torch.zeros(1)])
    
    print(f"Step 7: Concatenate ({len(tier_sigmas)} tier groups)")
    schedule = torch.cat(tier_sigmas)
    print(f"  Before sort: {[f'{float(v):.2f}' for v in schedule]}")
    schedule = torch.sort(schedule, descending=True).values
    print(f"  After sort: {[f'{float(v):.2f}' for v in schedule]}")
    schedule = torch.cat([schedule, torch.zeros(1)])
    print(f"  Final shape: {schedule.shape}")
    
    return schedule


def _karras_steps_in_range(n_steps, lo, hi, rho):
    """Generate Karras-polynomial steps within [lo, hi]."""
    if n_steps <= 0:
        return torch.tensor([], dtype=torch.float64)
    ramp = torch.linspace(0, 1, n_steps)
    lo_inv = lo ** (1 / rho)
    hi_inv = hi ** (1 / rho)
    sigmas = (hi_inv + ramp * (lo_inv - hi_inv)) ** rho
    sigmas = torch.sort(sigmas, descending=True).values
    return sigmas
