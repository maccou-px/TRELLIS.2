#!/bin/bash
# Full preprocessing pipeline for STL airplanes dataset.
# Must be run from: /home/jovyan/TRELLIS.2/trellis2/
#
# Usage:
#   cd /home/jovyan/TRELLIS.2/trellis2
#   bash ../process_planes.sh

set -e

# data_toolkit resolves the "datasets" plugin package from trellis2/
cd /home/jovyan/TRELLIS.2/trellis2

# DATASET_ROOT="/flux/vault/99_dev_martin/trellis_diffusion/datasets/planes"
# STL_DIR="/flux/vault/Conventional_Airplanes_geoms"
DATASET_ROOT="/flux/vault/99_dev_martin/trellis_diffusion/datasets/planes_subset"
STL_DIR="/flux/vault/Conventional_Airplanes_geoms_subset"

TRELLIS_TOOLKIT="/home/jovyan/TRELLIS.2/data_toolkit"

# Number of parallel workers for CPU-bound mesh dumping.
# Keep low to avoid OOM with Blender processes.
DUMP_WORKERS=8

# Dual grid input resolution fed to the shape encoder.
# 1024 matches the pretrained shape encoder's training distribution.
DUAL_GRID_RES=1024

# SS latent resolution (binary occupancy grid).
# Must match the ss encoder: ss_enc_conv3d_16l8_fp16_64 -> 64.
SS_RES=64

SHAPE_LATENT_NAME="shape_enc_next_dc_f16c32_fp16_${DUAL_GRID_RES}"
SS_LATENT_NAME="ss_enc_conv3d_16l8_fp16_${SS_RES}"

mkdir -p "$DATASET_ROOT"

# ---------------------------------------------------------------------------
echo "=== Step 1: Initialize metadata from STL files ==="
python "$TRELLIS_TOOLKIT/build_metadata.py" planes \
    --root "$DATASET_ROOT" \
    --stl_dir "$STL_DIR"

# ---------------------------------------------------------------------------
echo "=== Step 2: Dump meshes via Blender (STL -> .pickle) ==="
python "$TRELLIS_TOOLKIT/dump_mesh.py" planes \
    --root "$DATASET_ROOT" \
    --download_root "$STL_DIR" \
    --stl_dir "$STL_DIR" \
    --max_workers "$DUMP_WORKERS"

# ---------------------------------------------------------------------------
echo "=== Step 3: Update metadata after mesh dump ==="
python "$TRELLIS_TOOLKIT/build_metadata.py" planes \
    --root "$DATASET_ROOT"

# ---------------------------------------------------------------------------
echo "=== Step 4: Convert meshes to dual-grid O-Voxels (res ${DUAL_GRID_RES}) ==="
python "$TRELLIS_TOOLKIT/dual_grid.py" planes \
    --root "$DATASET_ROOT" \
    --resolution "$DUAL_GRID_RES"

# ---------------------------------------------------------------------------
echo "=== Step 5: Update metadata after dual grid ==="
python "$TRELLIS_TOOLKIT/build_metadata.py" planes \
    --root "$DATASET_ROOT"

# ---------------------------------------------------------------------------
echo "=== Step 6: Encode shape latents from dual grid ==="
python "$TRELLIS_TOOLKIT/encode_shape_latent.py" \
    --root "$DATASET_ROOT" \
    --resolution "$DUAL_GRID_RES"

# ---------------------------------------------------------------------------
echo "=== Step 7: Update metadata after shape latents ==="
python "$TRELLIS_TOOLKIT/build_metadata.py" planes \
    --root "$DATASET_ROOT"

# ---------------------------------------------------------------------------
echo "=== Step 8: Encode SS latents from shape latents (res ${SS_RES}) ==="
python "$TRELLIS_TOOLKIT/encode_ss_latent.py" \
    --root "$DATASET_ROOT" \
    --shape_latent_name "$SHAPE_LATENT_NAME" \
    --resolution "$SS_RES"

# ---------------------------------------------------------------------------
echo "=== Step 9: Final metadata update ==="
python "$TRELLIS_TOOLKIT/build_metadata.py" planes \
    --root "$DATASET_ROOT"

# ---------------------------------------------------------------------------
echo ""
echo "=== Dataset processing complete! ==="
echo "Dataset ready at: $DATASET_ROOT"
echo ""
echo "Train with:"
echo "  cd /home/jovyan/TRELLIS.2"
echo "  python train.py \\"
echo "    --config configs/gen/ss_flow_no_cond_1_3B_64_bf16.json \\"
echo "    --output_dir results/ss_flow_planes \\"
echo "    --data_dir '{\"planes\": {\"base\": \"${DATASET_ROOT}\", \"ss_latent\": \"${DATASET_ROOT}/ss_latents/${SS_LATENT_NAME}\"}}'"
