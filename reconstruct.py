# cf https://github.com/microsoft/TRELLIS.2/issues/53
import torch
import trimesh
import o_voxel
from trellis2 import models
from trellis2.modules.sparse import SparseTensor
from trellis2.representations import MeshWithVoxel


def load_shape_models():
    shape_enc = models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
    )
    shape_dec = models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"
    )
    for model in [shape_enc, shape_dec]:
        model.cuda().eval()

    return shape_enc, shape_dec


def normalize_mesh(mesh):
    """Normalize mesh vertices to unit cube."""
    vertices = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).long()

    center = (vertices.min(0)[0] + vertices.max(0)[0]) / 2
    scale = 0.99999 / (vertices.max(0)[0] - vertices.min(0)[0]).max()
    vertices = (vertices - center) * scale

    return vertices, faces


def encode_decode_shape(encoder, decoder, vertices, faces, resolution=512):
    voxel_indices, dual_vertices, intersected = (
        o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices,
            faces,
            grid_size=resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        )
    )

    # dual_vertices = dual_vertices * resolution - voxel_indices

    # add the batch dimension, 0 because we only have one mesh here!
    coords = torch.cat(
        [torch.zeros(len(voxel_indices), 1, dtype=torch.int32), voxel_indices], dim=1
    )
    vertices_sparse = SparseTensor(dual_vertices.cuda(), coords.cuda())
    intersected_sparse = SparseTensor(intersected.float().cuda(), coords.cuda())

    with torch.no_grad():
        latent = encoder(
            vertices_sparse, intersected_sparse
        )  # Downsampling happens here, TODO: understand why
        decoder.set_resolution(resolution)
        meshes, subs = decoder(latent, return_subs=True)

    return meshes[0], subs


def postprocess_with_default_texture(mesh, subs, resolution):
    pbr_attr_layout = {
        "base_color": slice(0, 3),
        "metallic": slice(3, 4),
        "roughness": slice(4, 5),
        "alpha": slice(5, 6),
    }

    print(f"[DEBUG] Input mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
    print(f"[DEBUG] Vertices bounds: {mesh.vertices.min(0)[0]} to {mesh.vertices.max(0)[0]}")
    mesh.fill_holes()
    print(f"[DEBUG] After fill_holes: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")

    # Create default white material
    sub = subs[-1]
    num_voxels = sub.coords.shape[0]
    default_attrs = torch.ones(
        num_voxels, 6, device=sub.coords.device, dtype=torch.float32
    )
    default_attrs[:, 3] = 0.0  # metallic
    default_attrs[:, 4] = 0.5  # roughness
    default_attrs[:, 5] = 1.0  # alpha

    mesh_with_voxel = MeshWithVoxel(
        mesh.vertices,
        mesh.faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1 / resolution,
        coords=sub.coords[:, 1:],
        attrs=default_attrs,
        voxel_shape=torch.Size([*sub.shape, *sub.spatial_shape]),
        layout=pbr_attr_layout,
    )

    mesh_with_voxel.simplify(16777216)
    print(f"[DEBUG] After simplify: {mesh_with_voxel.vertices.shape[0]} vertices, {mesh_with_voxel.faces.shape[0]} faces")

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh_with_voxel.vertices,
        faces=mesh_with_voxel.faces,
        attr_volume=mesh_with_voxel.attrs,
        coords=mesh_with_voxel.coords,
        attr_layout=mesh_with_voxel.layout,
        voxel_size=mesh_with_voxel.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000,
        texture_size=4096,
        remesh=True,
        geometry_only=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    return glb


def save_mesh(mesh, output_path):
    trimesh.Trimesh(
        vertices=mesh.vertices.cpu().numpy(),
        faces=mesh.faces.cpu().numpy(),
    ).export(output_path)


def main():
    shape_enc, shape_dec = load_shape_models()

    # mesh_path = "/home/jovyan/TRELLIS.2/data/plane.stl"
    mesh_path = "/home/jovyan/TRELLIS.2/data/bracket.stl"
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = list(mesh.geometry.values())[0]
    vertices, faces = normalize_mesh(mesh)

    resolution = 1024
    print("Encoding and decoding shape...")
    shape_mesh, subs = encode_decode_shape(
        shape_enc, shape_dec, vertices, faces, resolution=resolution
    )
    save_mesh(shape_mesh, "reconstructed_raw_shape.glb")

    print("Postprocessing mesh with default texture...")
    glb_mesh = postprocess_with_default_texture(shape_mesh, subs, resolution)
    glb_mesh.export("reconstructed_mesh_postprocessed.glb")


if __name__ == "__main__":
    main()
