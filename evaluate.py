"""
Evaluation script.

--- Mode 1: custom trained checkpoint ---
python evaluate.py --checkpoint /flux/vault/99_dev_martin/trellis_diffusion/results/big_planes_ss_flow_no_cond_1_3B_64_bf16/ckpts/denoiser_ema0.9999_step0142500.pt --config /flux/vault/99_dev_martin/trellis_diffusion/results/ss_flow_no_cond_1_3B_64_bf16/config.json --data_dir '{"ObjaverseXL_github": {"base": "datasets/ObjaverseXL_github", "ss_latent": "datasets/ObjaverseXL_github/ss_latents/ss_enc_conv3d_16l8_fp16_64"}}' --output eval_output/ --num_samples 50 --steps 50 --seed 42

--- Mode 2: pretrained Microsoft checkpoint (image-conditioned, run with null cond) ---
python evaluate.py --pretrained /flux/vault/pretrained_checkpoints/trellis/ss_flow_img_dit_1_3B_64_bf16 --ss_dec /flux/vault/pretrained_checkpoints/trellis/ss_dec_conv3d_16l8_fp16 --output eval_output_pretrained/ --num_samples 50 --steps 50
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

from trellis2 import models
from trellis2.models import SparseStructureFlowModel
from trellis2.pipelines.samplers import FlowEulerSampler
from trellis2.datasets.sparse_structure_latent import SparseStructureLatent


def load_model_from_config(
    config: dict, checkpoint_path: str
) -> SparseStructureFlowModel:
    """Load a custom-trained checkpoint using the training config."""
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


def load_model_from_pretrained(pretrained_path: str) -> SparseStructureFlowModel:
    """Load a pretrained checkpoint from .json + .safetensors files."""
    return models.from_pretrained(pretrained_path)


def build_dataset(config: dict, data_dir: str) -> SparseStructureLatent:
    """Instantiate SparseStructureLatent from training config for visualize_sample."""
    dataset_args = config["dataset"]["args"]
    return SparseStructureLatent(roots=data_dir, **dataset_args)


def build_dataset_pretrained(ss_dec_path: str) -> SparseStructureLatent:
    """Instantiate SparseStructureLatent using a local pretrained SS decoder."""
    return SparseStructureLatent(roots="{}", pretrained_ss_dec=ss_dec_path)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate SS flow model")
    # --- Mode 1: custom trained checkpoint ---
    parser.add_argument("--checkpoint", help="Path to .pt checkpoint (raw or EMA)")
    parser.add_argument(
        "--config", help="Training config JSON (results/<run>/config.json)"
    )
    parser.add_argument(
        "--data_dir", help="data_dir JSON string, same as used for training"
    )
    # --- Mode 2: pretrained safetensors checkpoint ---
    parser.add_argument(
        "--pretrained", help="Path to pretrained flow model (without extension)"
    )
    parser.add_argument(
        "--ss_dec", help="Path to pretrained SS decoder (without extension)"
    )
    # --- Common ---
    parser.add_argument("--output", default="eval_output")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    pretrained_mode = args.pretrained is not None

    if pretrained_mode:
        # --- Mode 2: pretrained ---
        assert args.ss_dec, "--ss_dec is required in pretrained mode"
        print(f"Loading pretrained model: {args.pretrained}")
        model = load_model_from_pretrained(args.pretrained).cuda().eval()
        sigma_min = 1e-5  # standard default
        print("Building dataset for visualization (pretrained decoder)...")
        dataset = build_dataset_pretrained(args.ss_dec)
    else:
        # --- Mode 1: custom trained checkpoint ---
        assert args.checkpoint and args.config and args.data_dir, (
            "--checkpoint, --config, and --data_dir are required in custom mode"
        )
        with open(args.config) as f:
            config = json.load(f)
        sigma_min = config["trainer"]["args"]["sigma_min"]
        print(f"Loading model: {args.checkpoint}")
        model = load_model_from_config(config, args.checkpoint).cuda().eval()
        print("Building dataset for visualization...")
        dataset = build_dataset(config, args.data_dir)

    R, C = model.resolution, model.in_channels
    cond_channels = getattr(model, "cond_channels", 0)
    print(f"  resolution={R}, in_channels={C}, cond_channels={cond_channels}")

    # --- Sample ---
    torch.manual_seed(args.seed)
    noise = torch.randn(args.num_samples, C, R, R, R, device="cuda")

    # Pretrained model is image-conditioned: pass null (zero) conditioning tokens
    if cond_channels > 0:
        cond = torch.zeros(args.num_samples, 1, cond_channels, device="cuda")
    else:
        cond = None

    sampler = FlowEulerSampler(sigma_min=sigma_min)
    print(f"Sampling {args.num_samples} shapes ({args.steps} steps)...")
    result = sampler.sample(
        model, noise=noise, cond=cond, steps=args.steps, verbose=True
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
