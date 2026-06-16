#!/usr/bin/env python3
"""
Minimal standalone example of the sd-flow sampler.

Demonstrates the FlowSigmaSchedule and FlowSampler with a dummy denoiser.

Usage:
    PYTHONPATH=src python examples/example.py
"""

import torch
from sd_flow import FlowSigmaSchedule, FlowSampler


def dummy_denoiser(x, sigma, **kwargs):
    """
    A toy denoiser that applies a simple noise-reduction heuristic.

    In a real Stable Diffusion pipeline this would be the UNet model.
    """
    # Simple heuristic: scale toward zero more at high sigma
    factor = torch.sigmoid(-sigma / 10.0)
    return x * factor


def main():
    print("sd-flow — Flow Sampler for Stable Diffusion")
    print("=" * 50)

    # ── Create a flow-based sigma schedule ──────────────────────────────
    print("\n1. Generating flow sigma schedule...")
    schedule = FlowSigmaSchedule(
        num_steps=10,
        sigma_min=0.002,
        sigma_max=80.0,
    )
    sigmas = schedule.generate_schedule()
    print(f"   Schedule shape: {list(sigmas.shape)}")
    print(f"   Sigmas: {[f'{float(v):.4f}' for v in sigmas]}")

    # ── Create the flow sampler (adaptive solver) ──────────────────────
    print("\n2. Creating flow sampler (adaptive flow solver)...")
    sampler = FlowSampler(solver="flow", schedule=schedule)

    # ── Run a dummy sampling loop ───────────────────────────────────────
    print("\n3. Running dummy sampling...")
    batch_size = 1
    channels = 4
    height = 8
    width = 8

    # Start from pure noise at sigma_max
    latents = torch.randn(batch_size, channels, height, width) * sigmas[0]

    result = sampler.sample(
        denoiser_fn=dummy_denoiser,
        latents=latents,
    )

    # The schedule used matches what was displayed above
    expected_len = len(sigmas) - 1  # num_steps
    assert sampler.schedule.num_steps == 10, "Example should use 10-step schedule"

    print(f"   Input shape:  {list(latents.shape)}")
    print(f"   Output shape: {list(result.shape)}")
    print(f"   Input  std:   {float(latents.std()):.4f}")
    print(f"   Output std:   {float(result.std()):.4f}")
    print(f"   Output min:   {float(result.min()):.4f}")
    print(f"   Output max:   {float(result.max()):.4f}")

    # ── Try Euler solver for comparison ─────────────────────────────────
    print("\n4. Trying Euler solver...")
    sampler_euler = FlowSampler(solver="euler")
    result_euler = sampler_euler.sample(
        denoiser_fn=dummy_denoiser,
        latents=torch.randn_like(latents) * sigmas[0],
    )
    print(f"   Euler output shape: {list(result_euler.shape)}")

    print("\n✅ Example completed successfully.")
    print("\n   Next steps:")
    print("   - Integrate into ComfyUI via integrations/comfyui/inject.sh")
    print("   - Or use FlowSigmaSchedule + FlowSampler in your own pipeline")


if __name__ == "__main__":
    main()
