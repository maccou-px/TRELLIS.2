# Pretrained Checkpoints

Location: `/flux/vault/pretrained_checkpoints/trellis/`

These are the Microsoft TRELLIS pretrained weights used by the pipeline. Each checkpoint has a `.json` (architecture config) and a `.safetensors` (weights) file.

---

## Pipeline position

```
Image → [ss_flow_img] → SS latent → [ss_dec] → occupancy/coords
                                                      │
                                    [shape_enc] ← mesh (preprocessing only)
                                                      │
                                               [slat_flow_512]
                                               [slat_flow_1024] → shape SLat → [shape_dec] → mesh
```

---

## Checkpoint reference

### `ss_dec_conv3d_16l8_fp16` (141 MB)
- **Class:** `SparseStructureDecoder`
- **Role:** Decodes the SS latent `[B, 8, 16, 16, 16]` → occupancy logits `[B, 1, 64, 64, 64]`
- **Architecture:** 3D conv VAE decoder (dense convolutions, no sparse ops), channels `[512, 128, 32]`
- **Used in:** Stage 1, both training and inference. Kept **frozen** during our SS flow training.
- **Precision:** fp16

---

### `ss_flow_img_dit_1_3B_64_bf16` (2.5 GB)
- **Class:** `SparseStructureFlowModel`
- **Role:** Image-conditioned flow model. Generates SS latents `[B, 8, 16, 16, 16]` from noise, conditioned on image tokens (`cond_channels=1024`)
- **Architecture:** Dense DiT, 1.3B params, 30 blocks, 12 heads, model_channels=1536, resolution=16
- **Used in:** Stage 1 at inference. **This is the Microsoft pretrained version** of what we are training from scratch (unconditionally).
- **Precision:** bfloat16

---

### `slat_flow_img2shape_dit_1_3B_512_bf16` (2.5 GB)
- **Class:** `SLatFlowModel`
- **Role:** First of two shape SLat flow models. Generates low-resolution shape latents (sparse, resolution=32) conditioned on image tokens
- **Architecture:** Sparse DiT, 1.3B params, 30 blocks, 12 heads, model_channels=1536, in/out channels=32
- **Used in:** Stage 2, first pass (LR). Runs sparse self-attention over K active voxel tokens.
- **Precision:** bfloat16

---

### `slat_flow_img2shape_dit_1_3B_1024_bf16` (2.5 GB)
- **Class:** `SLatFlowModel`
- **Role:** Second of two shape SLat flow models. Refines shape latents at higher resolution (resolution=64) after upsampling
- **Architecture:** Same as above but resolution=64 (operates on K' upsampled voxels)
- **Used in:** Stage 2, second pass (HR).
- **Precision:** bfloat16

---

### `shape_enc_next_dc_f16c32_fp16` (676 MB)
- **Class:** `FlexiDualGridVaeEncoder`
- **Role:** Encodes a mesh (via flexible dual grid representation) → shape SLat `[K, 32]`
- **Architecture:** Sparse UNet encoder, channels `[64, 128, 256, 512, 1024]`, latent_channels=32, SparseConvNeXt blocks
- **Used in:** Preprocessing only (extracting latents from training meshes). Not used at inference.
- **Precision:** fp16

---

### `shape_dec_next_dc_f16c32_fp16` (905 MB)
- **Class:** `FlexiDualGridVaeDecoder`
- **Role:** Decodes shape SLat `[K, 32]` → 7 channels per voxel (dual vertex offsets + intersected logits + quad_lerp), which are then converted to a mesh via `flexible_dual_grid_to_mesh()`
- **Architecture:** Sparse UNet decoder, channels `[1024, 512, 256, 128, 64]`, latent_channels=32, SparseConvNeXt blocks, resolution=256
- **Used in:** Stage 2 at inference to produce the final mesh.
- **Precision:** fp16

---

## What we train vs what we use pretrained

| Checkpoint | Status |
|---|---|
| `ss_dec_conv3d_16l8_fp16` | Pretrained, **frozen** |
| `ss_flow_img_dit_1_3B_64_bf16` | Pretrained (Microsoft image-conditioned version) |
| `slat_flow_img2shape_dit_1_3B_512_bf16` | Pretrained, used as-is |
| `slat_flow_img2shape_dit_1_3B_1024_bf16` | Pretrained, used as-is |
| `shape_enc_next_dc_f16c32_fp16` | Pretrained, used for preprocessing only |
| `shape_dec_next_dc_f16c32_fp16` | Pretrained, used as-is |
| **Our `denoiser`** (in `big_planes_ss_flow_no_cond_*`) | **Training from scratch** — unconditional SS flow |
