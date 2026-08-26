import torch
import numpy as np
from model import OceanSubsurfaceAutoencoder
from data_loader import load_data


def main():

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

    # Store predictions and actual values
    all_predictions = []
    all_actual = []

    with torch.no_grad():

        for X, Y in val_loader:

            X = X.to(device)
            Y = Y.to(device)

            prediction = model(X)

            all_predictions.append(prediction.cpu().numpy())
            all_actual.append(Y.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    actual = np.concatenate(all_actual, axis=0)

    print("\n--- R² Analysis ---")

    r2_scores = []

    # Calculate R² for each subsurface layer
    for layer in range(15):

        y_true = actual[:, layer, :, :].flatten()
        y_pred = predictions[:, layer, :, :].flatten()

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        r2 = 1 - (ss_res / ss_tot)

        r2_scores.append(r2)

        print(f"Layer {layer + 1:2d}: R² = {r2:.6f}")

    # Find best and worst layers
    best_layer = np.argmax(r2_scores) + 1
    worst_layer = np.argmin(r2_scores) + 1

    print("\n--- Results ---")
    print(f"Best layer based on R²: Layer {best_layer}")
    print(f"Worst layer based on R²: Layer {worst_layer}")

    print("\nR² analysis completed successfully!")


if __name__ == "__main__":
    main()