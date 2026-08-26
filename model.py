import torch
import torch.nn as nn
import torch.nn.functional as F

class SatelliteEncoder(nn.Module):
    def __init__(self, in_channels=5, latent_dim=128):
        super(SatelliteEncoder, self).__init__()
        
        # Layer 1: Expand 5 surface channels to 32 spatial feature maps
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Layer 2: Extract deeper patterns (32 -> 64 feature maps)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Layer 3: Bottleneck Compression (64 -> 128 latent dimensions)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        latent_embedding = self.block3(x)
        return latent_embedding


class SubsurfaceDecoder(nn.Module):
    def __init__(self, latent_dim=128, out_channels=15):
        super(SubsurfaceDecoder, self).__init__()
        
        # Layer 1: Transition down (128 -> 64 channels)
        self.block1 = nn.Sequential(
            nn.Conv2d(latent_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Layer 2: Transition down (64 -> 32 channels)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Layer 3: Final Projection to 15 Subsurface Layers
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        output = self.final_conv(x)
        return output

class OceanSubsurfaceAutoencoder(nn.Module):
    def __init__(self, in_channels=5, out_channels=15):
        super(OceanSubsurfaceAutoencoder, self).__init__()
        self.encoder = SatelliteEncoder(in_channels=in_channels, latent_dim=128)
        self.decoder = SubsurfaceDecoder(latent_dim=128, out_channels=out_channels)

    def forward(self, x):
        latent = self.encoder(x)
        predictions = self.decoder(latent)
        return predictions

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