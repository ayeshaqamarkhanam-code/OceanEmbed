import torch
import numpy as np

from model import OceanSubsurfaceAutoencoder
from data_loader import load_data


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    # Load validation data
    train_loader, val_loader = load_data()

    # Create model
    model = OceanSubsurfaceAutoencoder(
        in_channels=5,
        out_channels=15
    )

    # Load trained model
    model.load_state_dict(
        torch.load(
            "best_ocean_model.pth",
            map_location=device,
            weights_only=True
        )
    )

    model.to(device)
    model.eval()

    # Accumulate errors for all 15 layers
    mse_values = []
    mae_values = []

    print("\n--- Evaluating 15 Subsurface Layers ---")

    with torch.no_grad():

        for X, Y in val_loader:

            X = X.to(device)
            Y = Y.to(device)

            prediction = model(X)

            # Calculate error separately for each channel
            for layer in range(15):

                actual = Y[:, layer]
                predicted = prediction[:, layer]

                mse = torch.mean((actual - predicted) ** 2).item()
                mae = torch.mean(torch.abs(actual - predicted)).item()

                mse_values.append(mse)
                mae_values.append(mae)

    # Convert to arrays
    mse_values = np.array(mse_values).reshape(-1, 15)
    mae_values = np.array(mae_values).reshape(-1, 15)

    # Average over validation batches
    layer_mse = mse_values.mean(axis=0)
    layer_mae = mae_values.mean(axis=0)

    print("\nLayer-by-layer results:")
    print("--------------------------------")

    for layer in range(15):
        print(
            f"Layer {layer + 1:2d} | "
            f"MSE: {layer_mse[layer]:.6f} | "
            f"MAE: {layer_mae[layer]:.6f}"
        )

    print("--------------------------------")

    print("\nBest layer:")
    best_layer = np.argmin(layer_mae) + 1
    print(f"Layer {best_layer}")

    print("\nWorst layer:")
    worst_layer = np.argmax(layer_mae) + 1
    print(f"Layer {worst_layer}")

    print("\nLayer evaluation completed successfully!")


if __name__ == "__main__":
    main()