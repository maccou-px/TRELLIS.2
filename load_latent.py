import numpy as np


def main() -> None:
    latent_path = "/flux/vault/99_dev_martin/trellis_diffusion/datasets/planes/shape_latents/shape_enc_next_dc_f16c32_fp16_1024/fa439cf9274ea46431260d2f545a2d524fc02146ce14b3fab920175b1eca639f.npz"
    # load the latent
    latent = np.load(latent_path)

    print(latent)
    # print the shape of the latent
    print(latent.shape)


if __name__ == "__main__":
    main()
