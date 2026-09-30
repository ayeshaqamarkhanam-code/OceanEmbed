import torch
import torch.nn as nn
import torch.nn.functional as F


class SatelliteEncoder(nn.Module):
    """
    Compresses the 5-channel, 101x241 surface grid into a genuine
    spatial+channel embedding. Previous version only changed channel
    depth (32->64->128) with every conv at stride=1 — meaning it never
    actually compressed space, just relabeled each pixel. That's not
    an "embedding" in the sense the PS asks for, and gives the model
    far more free parameters per pixel than 8 training samples can
    support, encouraging overfitting.

    This version halves each spatial dimension twice (stride=2 convs),
    forcing the network to summarize a neighborhood of pixels into
    fewer numbers — a real bottleneck, and fewer effective parameters
    relative to the tiny dataset.
    """

    def __init__(self, in_channels=5, latent_dim=128):
        super(SatelliteEncoder, self).__init__()

        # Layer 1: Expand 5 surface channels to 32 spatial feature maps
        # (same resolution — 101x241)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Layer 2: downsample 101x241 -> 51x121, 32 -> 64 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Layer 3: downsample 51x121 -> 26x61, bottleneck to latent_dim.
        # This is the actual "satellite embedding" — (latent_dim, 26, 61)
        # instead of (latent_dim, 101, 241): a real compression, not a
        # per-pixel relabeling.
        self.block3 = nn.Sequential(
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU()
        )

        # Dropout kept ACTIVE at inference (model stays in .train() mode
        # for prediction calls) so repeated forward passes give slightly
        # different outputs. The spread across N passes is a real,
        # measurable uncertainty signal — Monte Carlo dropout. This is not
        # a cosmetic column: it is 0 only if the model is literally certain.
        self.mc_dropout = nn.Dropout2d(p=0.15)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.mc_dropout(x)
        latent_embedding = self.block3(x)
        return latent_embedding


class SubsurfaceDecoder(nn.Module):
    """
    Reconstructs 15 depth-temperature grids from the compressed latent
    embedding. Mirrors the encoder's two downsampling steps with two
    ConvTranspose2d upsampling steps back to 101x241. output_size is
    passed explicitly because 101 and 241 aren't exact multiples of the
    stride-2 downsampling, so PyTorch can't infer the exact size on its own.
    """

    def __init__(self, latent_dim=128, out_channels=15):
        super(SubsurfaceDecoder, self).__init__()

        # Layer 1: upsample 26x61 -> 51x121, latent_dim -> 64 channels
        self.up1 = nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Layer 2: upsample 51x121 -> 101x241, 64 -> 32 channels
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.relu = nn.ReLU()

        # Layer 3: Final Projection to 15 Subsurface Layers
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)

    def forward(self, x, out_hw=(101, 241)):
        b = x.shape[0]
        h1 = self.relu(self.bn1(self.up1(x, output_size=(b, 64, 51, 121))))
        h2 = self.relu(self.bn2(self.up2(h1, output_size=(b, 32, out_hw[0], out_hw[1]))))
        output = self.final_conv(h2)
        return output


class OceanSubsurfaceAutoencoder(nn.Module):
    def __init__(self, in_channels=5, out_channels=15):
        super(OceanSubsurfaceAutoencoder, self).__init__()
        self.encoder = SatelliteEncoder(in_channels=in_channels, latent_dim=128)
        self.decoder = SubsurfaceDecoder(latent_dim=128, out_channels=out_channels)

    def forward(self, x):
        latent = self.encoder(x)
        predictions = self.decoder(latent, out_hw=(x.shape[2], x.shape[3]))
        return predictions

    def predict_with_uncertainty(self, x, n_passes=20):
        """
        Monte Carlo dropout uncertainty. Runs n_passes forward passes with
        dropout kept ON, returns (mean, std) across those passes.
        std is a real per-pixel, per-depth uncertainty — not a placeholder.
        Call this instead of forward() wherever the app needs an "unc" value.
        """
        was_training = self.training
        self.train()  # keep dropout active even if caller is in eval mode
        preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                preds.append(self.forward(x))
        preds = torch.stack(preds, dim=0)  # (n_passes, B, 15, H, W)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        if not was_training:
            self.eval()
        return mean, std


if __name__ == "__main__":
    # Create a dummy batch representing: (Batch Size=2, Channels=5, Lat=101, Lon=241)
    dummy_input = torch.randn(2, 5, 101, 241)

    # Instantiate Model
    model = OceanSubsurfaceAutoencoder(in_channels=5, out_channels=15)

    # Run forward pass
    output = model(dummy_input)

    print("\n--- Model Verification ---")
    print(f"Input Shape:  {dummy_input.shape}")   # Expected: torch.Size([2, 5, 101, 241])
    print(f"Output Shape: {output.shape}")         # Expected: torch.Size([2, 15, 101, 241])
    print("Model architecture created successfully!")
