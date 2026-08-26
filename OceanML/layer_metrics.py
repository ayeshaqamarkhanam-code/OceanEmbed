import torch
import torch.nn as nn

from model import OceanSubsurfaceAutoencoder
from data_loader import load_data


def main():

    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    # Load data
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

    # Store errors for each of the 15 layers
    layer_mse = torch.zeros(15)
    layer_mae = torch.zeros(15)

    batches = 0

    print("\n--- Per-Layer Error Analysis ---")

    with torch.no_grad():

        for X, Y in val_loader:

            X = X.to(device)
            Y = Y.to(device)

            # Model prediction
            prediction = model(X)

            # Calculate error for each layer
            for layer in range(15):

                error = prediction[:, layer] - Y[:, layer]

                mse = torch.mean(error ** 2)
                mae = torch.mean(torch.abs(error))

                layer_mse[layer] += mse.cpu()
                layer_mae[layer] += mae.cpu()

            batches += 1

    # Average across batches
    layer_mse /= batches
    layer_mae /= batches

    # Print results
    print("\nLayer        MSE              MAE")
    print("-----------------------------------------")

    for layer in range(15):

        print(
            f"{layer + 1:5d}   "
            f"{layer_mse[layer].item():.6f}   "
            f"{layer_mae[layer].item():.6f}"
        )

    # Find best and worst layers
    best_mse_layer = torch.argmin(layer_mse).item() + 1
    worst_mse_layer = torch.argmax(layer_mse).item() + 1

    best_mae_layer = torch.argmin(layer_mae).item() + 1
    worst_mae_layer = torch.argmax(layer_mae).item() + 1

    print("\n--- Results ---")

    print(
        f"Best layer based on MSE: Layer {best_mse_layer}"
    )

    print(
        f"Worst layer based on MSE: Layer {worst_mse_layer}"
    )

    print(
        f"Best layer based on MAE: Layer {best_mae_layer}"
    )

    print(
        f"Worst layer based on MAE: Layer {worst_mae_layer}"
    )

    print("\nLayer analysis completed successfully!")


if __name__ == "__main__":
    main()