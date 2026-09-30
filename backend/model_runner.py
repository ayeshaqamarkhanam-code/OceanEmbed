import os
import sys
import numpy as np
import torch

# Allow importing the model from OceanML
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OCEANML_DIR = os.path.join(PROJECT_ROOT, "OceanML")
sys.path.insert(0, OCEANML_DIR)

from model import OceanSubsurfaceAutoencoder


DEPTHS = [
    0, 5, 10, 20, 30,
    50, 75, 100, 125, 150,
    200, 300, 500, 700, 1000
]

MODEL_PATH = os.path.join(OCEANML_DIR, "best_ocean_model.pth")
X_PATH = os.path.join(OCEANML_DIR, "INCOIS_X_surface.npy")
Y_PATH = os.path.join(OCEANML_DIR, "INCOIS_Y_subsurface.npy")

DEVICE = torch.device("cpu")

# Load model
model = OceanSubsurfaceAutoencoder(
    in_channels=5,
    out_channels=15
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.to(DEVICE)
model.eval()

# Load dataset
X_DATA = np.load(X_PATH)
Y_DATA = np.load(Y_PATH)

# Same normalization used during training
X_TENSOR = torch.tensor(X_DATA, dtype=torch.float32)
Y_TENSOR = torch.tensor(Y_DATA, dtype=torch.float32)

MEAN_X = X_TENSOR.mean(dim=(0, 2, 3), keepdim=True)
STD_X = X_TENSOR.std(
    dim=(0, 2, 3),
    keepdim=True,
    unbiased=False
).clamp(min=1e-8)

MEAN_Y = Y_TENSOR.mean(dim=(0, 2, 3), keepdim=True)
STD_Y = Y_TENSOR.std(
    dim=(0, 2, 3),
    keepdim=True,
    unbiased=False
).clamp(min=1e-8)


def predict(day_index: int):
    """
    Run the trained OceanEmbed model for one dataset day.
    Returns (mean, std) — std is real Monte Carlo dropout uncertainty,
    in the same de-normalized temperature units as mean.
    """

    if not 0 <= day_index < len(X_DATA):
        raise ValueError(
            f"day_index must be between 0 and {len(X_DATA) - 1}"
        )

    x = X_TENSOR[day_index:day_index + 1]

    # Normalize exactly like training
    x = (x - MEAN_X) / STD_X

    mean_pred, std_pred = model.predict_with_uncertainty(x.to(DEVICE), n_passes=20)

    # Convert back to original temperature scale.
    # mean: de-normalize normally. std: only rescale (no mean shift —
    # std of a shifted variable is unaffected by the shift, only the scale).
    mean_pred = mean_pred * STD_Y + MEAN_Y
    std_pred = std_pred * STD_Y

    return mean_pred.squeeze(0).cpu().numpy(), std_pred.squeeze(0).cpu().numpy()


def get_depth_profile(
    day_index: int,
    latitude: float,
    longitude: float
):
    """
    Get the 15-depth temperature profile at the nearest
    0.25-degree grid point.
    """

    mean_pred, std_pred = predict(day_index)

    lat_index = round((latitude - 5.0) / 0.25)
    lon_index = round((longitude - 45.0) / 0.25)

    if not 0 <= lat_index < 101:
        raise ValueError(
            "Latitude is outside the North Indian Ocean grid."
        )

    if not 0 <= lon_index < 241:
        raise ValueError(
            "Longitude is outside the North Indian Ocean grid."
        )

    if X_DATA[day_index, 0, lat_index, lon_index] == 0:
        raise ValueError(
            "Selected point is on land in this dataset's grid. Pick an ocean point."
        )

    temperatures = mean_pred[:, lat_index, lon_index]
    uncertainties = std_pred[:, lat_index, lon_index]

    depth_profile = [
        {
            "depth": depth,
            "temp": float(temperatures[i]),
            "unc": float(uncertainties[i])
        }
        for i, depth in enumerate(DEPTHS)
    ]

    return {
        "latitude": 5.0 + lat_index * 0.25,
        "longitude": 45.0 + lon_index * 0.25,
        "depth_profile": depth_profile
    }