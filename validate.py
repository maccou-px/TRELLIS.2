"""
Validation script for trained SparseStructureFlowModel checkpoints.

Usage:
    python validate.py \
        --checkpoint results/ss_flow_img_dit_1_3B_64_bf16/ckpts/denoiser_ema0.9999_step0000100.pt \
        --image assets/example_image/T.png \
        --output validation_output/

Optional args:
    --config configs/gen/ss_flow_img_dit_1_3B_64_bf16.json  (default)
    --pipeline microsoft/TRELLIS.2-4B                        (HF repo for full pipeline)
    --seed 42
    --num_steps 50                                           (sampler steps, overrides pipeline default)
    --pipeline_type 512                                      (512 | 1024 | 1024_cascade)
"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import torch
import imageio
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'trellis2'))

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.models import SparseStructureFlowModel
from trellis2.utils import render_utils


def load_trained_model(config_path: str, checkpoint_path: str) -> SparseStructureFlowModel:
    """Build SparseStructureFlowModel from training config and load checkpoint weights."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    model_args = config['models']['denoiser']['args']
    model = SparseStructureFlowModel(**model_args)

    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print(f"  Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # Convert to bfloat16 so FlashAttention works without needing torch.autocast.
    # The pretrained pipeline models are bf16; our checkpoint is fp32 by default.
    model.convert_to(torch.bfloat16)

    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description='Validate a trained ss_flow checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .pt checkpoint file (e.g. denoiser_ema0.9999_step0000100.pt)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', type=str, default='validation_output',
                        help='Output directory')
    parser.add_argument('--config', type=str,
                        default='configs/gen/ss_flow_img_dit_1_3B_64_bf16.json',
                        help='Training config JSON')
    parser.add_argument('--pipeline', type=str,
                        default='microsoft/TRELLIS.2-4B',
                        help='Pretrained pipeline HF repo or local path')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_steps', type=int, default=50,
                        help='Number of diffusion sampler steps')
    parser.add_argument('--pipeline_type', type=str, default='512',
                        choices=['512', '1024', '1024_cascade', '1536_cascade'],
                        help='Pipeline type (use 512 for fastest validation)')
    parser.add_argument('--no_glb', action='store_true',
                        help='Skip GLB export (faster)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 1. Load full pretrained pipeline
    print(f"Loading pretrained pipeline from: {args.pipeline}")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.pipeline)
    pipeline.cuda()

    # 2. Build and load trained sparse structure flow model (in bf16)
    print("Loading trained model...")
    trained_model = load_trained_model(args.config, args.checkpoint)
    trained_model = trained_model.cuda()

    # 3. Swap sparse_structure_flow_model in the pipeline
    pipeline.models['sparse_structure_flow_model'] = trained_model
    print("Replaced sparse_structure_flow_model with trained checkpoint.")

    # 4. Load and preprocess input image
    image = Image.open(args.image)
    print(f"Input image: {args.image} ({image.size})")

    # 5. Run pipeline (no autocast needed — model is already bf16)
    print(f"Running pipeline (type={args.pipeline_type}, seed={args.seed}, steps={args.num_steps})...")
    sampler_params = {'steps': args.num_steps}
    meshes = pipeline.run(
        image,
        seed=args.seed,
        sparse_structure_sampler_params=sampler_params,
        pipeline_type=args.pipeline_type,
    )
    mesh = meshes[0]
    print("Pipeline complete.")

    # 6. Render video
    print("Rendering video...")
    video_frames = render_utils.render_video(mesh, num_frames=60, resolution=512)
    video_path = os.path.join(args.output, 'output.mp4')
    imageio.mimsave(video_path, video_frames['color'], fps=15)
    print(f"Video saved: {video_path}")

    # 7. Export GLB
    if not args.no_glb:
        try:
            import o_voxel
            print("Exporting GLB...")
            mesh.simplify(16777216)
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=500000,
                texture_size=1024,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=True,
            )
            glb_path = os.path.join(args.output, 'output.glb')
            glb.export(glb_path)
            print(f"GLB saved: {glb_path}")
        except Exception as e:
            print(f"GLB export failed: {e}")

    print(f"\nDone. Results in: {args.output}/")


if __name__ == '__main__':
    main()
