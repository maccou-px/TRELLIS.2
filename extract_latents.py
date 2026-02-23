# cf https://github.com/microsoft/TRELLIS.2/issues/53
from pathlib import Path
import torch
import trimesh
import o_voxel
from trellis2 import models
from trellis2.modules.sparse import SparseTensor


def load_shape_models():
    shape_enc = models.from_pretrained(
        # Next-DC block type (the sparse ConvNeXt-style blocks used in the UNet)
        "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
    )
    for model in [shape_enc]:
        model.cuda().eval()

    return shape_enc


shape_enc = load_shape_models()
resolution = 1024


def normalize_mesh(mesh):
    """Normalize mesh vertices to unit cube."""
    vertices = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).long()

    bounding_box_min = vertices.min(dim=0).values
    bounding_box_max = vertices.max(dim=0).values
    center = (bounding_box_max + bounding_box_min) / 2.0
    scale = 0.99999 / (bounding_box_max - bounding_box_min).max()

    vertices = (vertices - center) * scale

    return vertices, faces


def encode_shape(encoder, vertices, faces, resolution=512):
    # voxel indices are 0-indexed.
    voxel_indices, dual_vertices, intersected = (
        o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices,
            faces,
            grid_size=resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        )
    )

    # Convert dual_vertices from world-normalized [0,1] space to voxel-local [0,1] space.
    # Example with resolution=10: voxel_indices=[5,3,7], dual_vertices=[0.52, 0.31, 0.73]
    #   dual_vertices * 10 = [5.2, 3.1, 7.3]  (fractional grid position)
    #   - voxel_indices    = [0.2, 0.1, 0.3]  (offset within the voxel)
    # This removes positional bias so the model can focus on local surface geometry rather than global position.
    dual_vertices = dual_vertices * resolution - voxel_indices

    # add the batch dimension, 0 because we only have one mesh here!
    coords = torch.cat(
        [torch.zeros(len(voxel_indices), 1, dtype=torch.int32), voxel_indices], dim=1
    )
    vertices_sparse = SparseTensor(dual_vertices.cuda(), coords.cuda())
    intersected_sparse = SparseTensor(intersected.float().cuda(), coords.cuda())

    with torch.no_grad():
        latent = encoder(
            vertices_sparse, intersected_sparse
        )  # Downsampling happens here.

    return latent.cpu()


def save_latent(latent, output_path):
    torch.save(latent, output_path)


def run_one(sample_name: str):
    export_path = (
        Path("/flux/vault/99_dev_martin/trellis_latents") / f"{sample_name}_latent.pt"
    )
    if export_path.exists():
        print(f"Reconstructed mesh already exists at {export_path}, skipping...")
        return
    mesh_path = f"/flux/vault/Conventional_Airplanes_geoms/{sample_name}.stl"

    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = list(mesh.geometry.values())[0]
    vertices, faces = normalize_mesh(mesh)

    print("Encoding shape...")
    latent = encode_shape(shape_enc, vertices, faces, resolution=resolution)
    save_latent(latent, export_path)


def main():
    # run_one("sample_20242312_1")
    data_path = Path("/flux/vault/Conventional_Airplanes_geoms")
    for sample_path in data_path.glob("*.stl"):
        sample = sample_path.stem
        print(f"Processing sample: {sample}")
        run_one(sample)


if __name__ == "__main__":
    main()
