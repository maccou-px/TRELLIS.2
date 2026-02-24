# Full TRELLIS Generative Pipeline

```
INPUT IMAGE  [H, W, 4]  (RGBA, preprocessed to ~518×518)
     │
     ▼  ViT/DINO image encoder
COND TOKENS  [N, 1024]   (N = num image patches ≈ 1024)
     │
     ├──────────────────────────────────────────────────────────────
     │               STAGE 1 — Sparse Structure (SS)
     ├──────────────────────────────────────────────────────────────
     │
     ▼  draw from N(0,I)
NOISE        [B, 8, 16, 16, 16]   ← dense 3D latent space
     │
     ▼  SparseStructureFlowModel (Dense DiT, 50 Euler steps)
     │    each step:  model(x_t, t*1000, cond) → velocity v
     │                x_{t-1} = x_t - Δt * v
SS LATENT z_s [B, 8, 16, 16, 16]
     │
     ▼  SparseStructureDecoder  (3D conv VAE decoder — dense convolutions only, NO o_voxel)
OCCUPANCY    [B, 1, 64, 64, 64]   (logits, threshold at 0)
     │
     ▼  argwhere(decoded > 0)
COORDS       [K, 4]   (batch_idx, x, y, z)   K ≈ 2k–8k active voxels
     │
     ├──────────────────────────────────────────────────────────────
     │               STAGE 2 — Shape SLat  (Sparse)
     │               *** o_voxel flexible dual grid used here ***
     ├──────────────────────────────────────────────────────────────
     │
     │  NOTE: shape latents were produced during preprocessing by:
     │    mesh → mesh_to_flexible_dual_grid()
     │         → (voxel_indices [K,3], dual_vertices [K,3], intersected [K,3])
     │         → concat [K, 6] → FlexiDualGridVaeEncoder → SparseTensor([K, 32])
     │
     ▼  draw from N(0,I) at each active voxel position
NOISE  SparseTensor(feats=[K, 32], coords=[K, 4])   ← 512-grid LR pass
     │
     ▼  ShapeSLatFlowModel_512 (Sparse DiT, 50 Euler steps)
     │    each step runs sparse self-attn over K tokens
SHAPE_SLAT_LR  SparseTensor(feats=[K, 32], coords=[K, 4])
     │
     ▼  shape_decoder.upsample(×4)  → predicts sub-voxel offsets
HR_COORDS  [K', 4]   at 1024 grid   K' ≈ 10k–50k
     │
     ▼  draw from N(0,I) at K' positions
NOISE  SparseTensor(feats=[K', 32], coords=[K', 4])
     │
     ▼  ShapeSLatFlowModel_1024 (Sparse DiT, 50 Euler steps)
SHAPE_SLAT  SparseTensor(feats=[K', 32], coords=[K', 4])
     │  (de-normalized: slat = slat * std + mean)
     │
     ▼  FlexiDualGridVaeDecoder  (Sparse UNet → 7 channels per voxel)
     │    [0:3] → dual vertex offsets   [3:6] → intersected logits   [6] → quad_lerp
     ▼  o_voxel.convert.flexible_dual_grid_to_mesh(coords, vertices, intersected, quad_lerp)
MESH         vertices=[V, 3],  faces=[F, 3]    V ≈ 100k–500k
SUBS[-1]     SparseTensor(feats=[K'', 32])     finest-res sparse grid
     │
     ├──────────────────────────────────────────────────────────────
     │               STAGE 3 — Texture SLat  (Sparse)
     ├──────────────────────────────────────────────────────────────
     │
     ▼  shape_slat (normalized) ++ noise [K', 32]  →  concat → [K', 64]
NOISE  SparseTensor(feats=[K', 64], coords=[K', 4])
     │
     ▼  TexSLatFlowModel_1024 (Sparse DiT, 50 Euler steps)
     │    concat_cond = shape_slat fed as extra channels
TEX_SLAT     SparseTensor(feats=[K', 6], coords=[K', 4])
     │  6 channels = (R, G, B, metallic, roughness, alpha)
     │
     ▼  TexSLatDecoder  (Sparse UNet + subs guidance)
PBR_VOXELS   SparseTensor(feats=[K'', 6])   at fine 1024 grid
     │
     ├──────────────────────────────────────────────────────────────
     │               POST-PROCESS → GLB
     ├──────────────────────────────────────────────────────────────
     │
     ▼  MeshWithVoxel  (mesh + voxel PBR attributes)
     ▼  o_voxel.postprocess.to_glb()
     │    - hole filling + decimation (or optional Dual Contouring remesh)
     │    - UV unwrapping
     │    - PBR attribute baking from voxel volume to UV texture
OUTPUT GLB   textured 3D mesh
```

---

## How data flows through the Sparse UNet decoder (FlexiDualGridVaeDecoder)

### What a SparseTensor actually is

Forget "3D volume". A `SparseTensor` is just two tables:

```
coords  [K, 4]   — integer (batch_idx, x, y, z) for each active voxel
feats   [K, C]   — C floats of data for each active voxel
```

`K` is whatever it is. Nothing is padded to a fixed grid size.

### Step-by-step through SparseUnetVaeDecoder.forward()

Say K=5000 voxels come in at a coarse resolution (e.g. 128³ grid), with 32 latent channels.

**1. `from_latent` — linear projection** (`sparse_unet_vae.py:482`)
```
feats:  [5000, 32]  →  [5000, 64]     # Linear on each row independently
coords: [5000, 4]   unchanged
```

**2. Several conv blocks — process in place**
```
feats:  [5000, 64]  →  [5000, 64]     # sparse 3×3×3 conv gathers neighbors
coords: [5000, 4]   unchanged          # K stays the same
```

**3. `SparseResBlockUpsample3d` — where K grows** (`sparse_unet_vae.py:131`)

1. `to_subdiv`: linear → `[5000, 8]` — one logit per sub-voxel child (each voxel has 8 children in a 2× finer grid)
2. Binarize: which children are "on"? Say avg 4 of 8 → K' ≈ 20000
3. `SparseUpsample(2)`: multiply coords × 2, add sub-voxel offset for each surviving child

```
# spatial/basic.py:94-97
new_coords[:, 1:] *= self.factor       # scale up
new_coords[...] += subidx_offset       # add child offset (0,0,0) (0,0,1) (0,1,0) ...
```

Result:
```
feats:  [5000, 64]  →  [20000, 64]    # each child copies parent features
coords: [5000, 4]   →  [20000, 4]    # at 2× finer resolution
```
The network decided which voxels to expand, not the caller.

**4. More conv blocks at fine resolution**
```
feats:  [20000, 64]  →  [20000, 32]
coords: [20000, 4]   unchanged
```

**5. `output_layer` — final linear projection** (`sparse_unet_vae.py:500`)
```
feats:  [20000, 32]  →  [20000, 7]
coords: [20000, 4]   unchanged
```

**6. Back in `FlexiDualGridVaeDecoder.forward()` — split the 7 channels** (`fdg_vae.py:87-89`)
```python
vertices           = sigmoid(feats[..., 0:3])   # [20000, 3]  sub-voxel vertex offsets
intersected_logits = feats[..., 3:6]            # [20000, 3]  which edges cross surface
quad_lerp          = softplus(feats[..., 6:7])  # [20000, 1]  blending weight
```

**7. `flexible_dual_grid_to_mesh(coords, vertices, intersected, quad_lerp, grid_size=resolution)`**

`resolution` is just the denominator that converts integer coords to world-space positions in `[-0.5, 0.5]³`. It does not constrain the network.

### Why different input sizes work

Every operation is either:
- A **Linear on rows**: `[K, C_in] → [K, C_out]` — K is irrelevant
- A **sparse conv**: gathers neighbors from the coord table — works for any K
- A **coord arithmetic step**: multiply/add integers — works for any K

There is no reshape to a fixed grid anywhere. K flows through as the row dimension.

---

## What we are training (unconditioned SS flow)

Only Stage 1 is trained, with `cond=None`:

```
NOISE   [B, 8, 16, 16, 16]   ← pure N(0,I)
  │
  ▼  SparseStructureFlowModel (no conditioning)
z_s     [B, 8, 16, 16, 16]   ← SS latent
  │
  ▼  SparseStructureDecoder (pretrained, frozen)
OCCUPANCY [B, 1, 64, 64, 64] → binary → voxel shape visualization
```

The model learns the distribution of SS latents over the airplane dataset.
Stages 2 and 3 (shape/texture SLat) are not used during training — those remain the pretrained Microsoft models.

---

## Key shape summary

| Variable | Shape | What it is |
|---|---|---|
| SS latent z_s | `[B, 8, 16, 16, 16]` | Dense 3D latent, the diffusion target |
| Occupancy grid | `[B, 1, 64, 64, 64]` | Binary voxel presence |
| Coords (sparse) | `[K, 4]` | Active voxel positions, K varies |
| Shape SLat feats | `[K, 32]` | Per-voxel geometry features |
| Tex SLat feats | `[K', 6]` | Per-voxel PBR attributes |
| Final mesh | `[V,3] + [F,3]` | Flexible Dual Grid surface (from Stage 2) |
