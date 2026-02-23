import torch


def main() -> None:
    latent_path = (
        "/flux/vault/99_dev_martin/trellis_latents/sample_20242312_1_latent.pt"
    )
    latent = torch.load(latent_path, weights_only=False)
    print(latent)


if __name__ == "__main__":
    main()
