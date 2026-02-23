"""
Evaluation script for the unconditioned SparseStructureFlowModel.

Mirrors what FlowMatchingTrainer.run_snapshot + BasicTrainer.snapshot do,
but as a standalone script: sample from pure noise, then decode and render
via SparseStructureLatent.visualize_sample.

Usage:
    python evaluate_unconditioned.py \
        --checkpoint results/planes_ss_flow_no_cond_1_3B_64_bf16/ckpts/denoiser_ema0.9999_step0000100.pt \
        --config     results/ss_flow_no_cond_1_3B_64_bf16/config.json \
        --data_dir "{\"ObjaverseXL_github\": {\"base\": \"datasets/ObjaverseXL_github\", \"ss_latent\": \"datasets/ObjaverseXL_github/ss_latents/ss_enc_conv3d_16l8_fp16_64\"}}"
        --output     eval_output/ \
        --num_samples 16 \
        --steps 50 \
        --seed 42
"""

import os
import argparse
import json
import math

import numpy as np
import torch
import torchvision
from PIL import Image

import sys

sys.path.insert(0, os.path.dirname(__file__))

from trellis2.models import SparseStructureFlowModel
from trellis2.pipelines.samplers import FlowEulerSampler
from trellis2.datasets.sparse_structure_latent import SparseStructureLatent


def load_model(config: dict, checkpoint_path: str) -> SparseStructureFlowModel:
    args = config["models"]["denoiser"]["args"]
    model = SparseStructureFlowModel(**args)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:
        print(f"  [warn] missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"  [warn] unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
    model.convert_to(torch.bfloat16)
    return model


def build_dataset(config: dict, data_dir: str) -> SparseStructureLatent:
    """Instantiate SparseStructureLatent from training config for visualize_sample."""
    dataset_args = config["dataset"]["args"]
    return SparseStructureLatent(roots=data_dir, **dataset_args)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate unconditioned SS flow model")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to .pt checkpoint (raw or EMA)"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Training config JSON (results/<run>/config.json)",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="data_dir JSON string, same as used for training",
    )
    parser.add_argument("--output", default="eval_output")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(args.config) as f:
        config = json.load(f)

    sigma_min = config["trainer"]["args"]["sigma_min"]

    # --- Model ---
    print(f"Loading model: {args.checkpoint}")
    model = load_model(config, args.checkpoint).cuda().eval()
    R, C = model.resolution, model.in_channels
    print(f"  resolution={R}, in_channels={C}")

    # --- Dataset (only needed for visualize_sample / decode_latent) ---
    print("Building dataset for visualization...")
    dataset = build_dataset(config, args.data_dir)

    # --- Sample ---
    torch.manual_seed(args.seed)
    noise = torch.randn(args.num_samples, C, R, R, R, device="cuda")

    sampler = FlowEulerSampler(sigma_min=sigma_min)
    print(f"Sampling {args.num_samples} shapes ({args.steps} steps)...")
    result = sampler.sample(
        model, noise=noise, cond=None, steps=args.steps, verbose=True
    )
    samples = result.samples  # [N, C, R, R, R]

    # --- Decode + Render (reuses SparseStructureLatentVisMixin.visualize_sample) ---
    print("Decoding and rendering...")
    images = dataset.visualize_sample(samples)  # [N, 3, 1024, 1024]

    # --- Save grid ---
    nrow = math.ceil(math.sqrt(args.num_samples))
    grid_path = os.path.join(args.output, "grid.png")
    torchvision.utils.save_image(
        images, grid_path, nrow=nrow, normalize=True, value_range=(0, 1)
    )
    print(f"Grid saved: {grid_path}")

    # --- Save individual renders ---
    for i in range(images.shape[0]):
        img = images[i].permute(1, 2, 0).cpu().float().numpy().clip(0, 1)
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(args.output, f"sample_{i:04d}.png"))

    print(f"\nDone. {args.num_samples} samples saved to: {args.output}/")


if __name__ == "__main__":
    main()
