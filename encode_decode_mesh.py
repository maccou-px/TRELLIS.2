"""
Complete workflow for encoding a mesh to latent space and decoding it back.

This script demonstrates:
1. Loading a mesh
2. Converting mesh to O-Voxel representation
3. Encoding O-Voxel to latent space using the shape encoder
4. Decoding latent back to O-Voxel using the shape decoder
5. Converting O-Voxel back to mesh
"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import trimesh
import o_voxel
from trellis2.pipelines import Trellis2TexturingPipeline
from trellis2.modules.sparse import SparseTensor

# Configuration
INPUT_MESH = "path/to/your/mesh.glb"  # Change this to your mesh path
RESOLUTION = 512  # Voxel grid resolution (512, 1024, or 1536)
OUTPUT_MESH = "reconstructed_mesh.glb"

print("=" * 60)
print("TRELLIS.2 Mesh Encoding/Decoding Demo")
print("=" * 60)

# ============================================================================
# Step 1: Load the pretrained models
# ============================================================================
print("\n[1/6] Loading pretrained models...")
pipeline = Trellis2TexturingPipeline.from_pretrained(
    "microsoft/TRELLIS.2-4B",
    config_file="texturing_pipeline.json"
)
pipeline.cuda()

# Extract the encoder and decoder
shape_encoder = pipeline.models['shape_slat_encoder']
shape_decoder = pipeline.models['shape_slat_decoder']

print(f"✓ Encoder loaded: {type(shape_encoder).__name__}")
print(f"✓ Decoder loaded: {type(shape_decoder).__name__}")

# ============================================================================
# Step 2: Load and normalize the input mesh
# ============================================================================
print(f"\n[2/6] Loading mesh from: {INPUT_MESH}")
asset = trimesh.load(INPUT_MESH)

# Normalize to unit cube [-0.5, 0.5]
aabb = asset.bounding_box.bounds
center = (aabb[0] + aabb[1]) / 2
scale = 0.99999 / (aabb[1] - aabb[0]).max()
asset.apply_translation(-center)
asset.apply_scale(scale)

print(f"✓ Mesh loaded with {len(asset.vertices) if hasattr(asset, 'vertices') else 'N/A'} vertices")

# ============================================================================
# Step 3: Convert mesh to O-Voxel representation
# ============================================================================
print(f"\n[3/6] Converting mesh to O-Voxel (resolution: {RESOLUTION}³)...")

# Geometry voxelization
mesh = asset.to_mesh() if hasattr(asset, 'to_mesh') else asset
vertices = torch.from_numpy(mesh.vertices).float()
faces = torch.from_numpy(mesh.faces).long()

voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
    vertices, faces,
    grid_size=RESOLUTION,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    face_weight=1.0,
    boundary_weight=0.2,
    regularization_weight=1e-2,
    timing=True
)

# Sort voxels
vid = o_voxel.serialize.encode_seq(voxel_indices)
mapping = torch.argsort(vid)
voxel_indices = voxel_indices[mapping]
dual_vertices = dual_vertices[mapping]
intersected = intersected[mapping]

print(f"✓ Voxelization complete: {len(voxel_indices)} occupied voxels")

# ============================================================================
# Step 4: Encode O-Voxel to latent space
# ============================================================================
print(f"\n[4/6] Encoding O-Voxel to latent space...")

# Prepare input tensors for encoder
vertices_sparse = SparseTensor(
    feats=dual_vertices * RESOLUTION - voxel_indices,
    coords=torch.cat([torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1)
).to('cuda')

intersected_sparse = vertices_sparse.replace(intersected).to('cuda')

# Encode
with torch.no_grad():
    shape_latent = shape_encoder(vertices_sparse, intersected_sparse)

print(f"✓ Encoding complete!")
print(f"  - Latent shape: {shape_latent.feats.shape}")
print(f"  - Latent coords: {shape_latent.coords.shape}")
print(f"  - Compression ratio: {len(voxel_indices) / len(shape_latent.coords):.2f}x")

# ============================================================================
# Step 5: Decode latent back to O-Voxel
# ============================================================================
print(f"\n[5/6] Decoding latent back to O-Voxel...")

shape_decoder.set_resolution(RESOLUTION)
with torch.no_grad():
    decoded = shape_decoder(shape_latent, return_subs=True)
    decoded_voxels, decoded_subs = decoded

print(f"✓ Decoding complete!")
print(f"  - Decoded voxels: {len(decoded_voxels.coords)}")

# ============================================================================
# Step 6: Convert O-Voxel back to mesh and export
# ============================================================================
print(f"\n[6/6] Converting O-Voxel back to mesh...")

# Extract mesh from decoded voxels
coords = decoded_voxels.coords[:, 1:].cpu()
feats = decoded_voxels.feats.cpu()

# Split features into dual vertices and intersected flags
dual_verts_decoded = feats[:, :3]
intersected_decoded = feats[:, 3:].bool()

# Reconstruct mesh
rec_vertices, rec_faces = o_voxel.convert.flexible_dual_grid_to_mesh(
    coords.cuda(),
    dual_verts_decoded.cuda(),
    intersected_decoded.cuda(),
    split_weight=None,
    grid_size=RESOLUTION,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
)

# Create trimesh object
reconstructed_mesh = trimesh.Trimesh(
    vertices=rec_vertices.cpu().numpy(),
    faces=rec_faces.cpu().numpy()
)

# Export
reconstructed_mesh.export(OUTPUT_MESH)

print(f"✓ Mesh exported to: {OUTPUT_MESH}")
print(f"  - Vertices: {len(rec_vertices)}")
print(f"  - Faces: {len(rec_faces)}")

print("\n" + "=" * 60)
print("✓ Encoding/Decoding complete!")
print("=" * 60)
