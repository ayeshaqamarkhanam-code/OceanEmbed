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
    all_masks = []

    with torch.no_grad():

        for X, Y, mask in val_loader:

            X = X.to(device)
            Y = Y.to(device)

            prediction = model(X)

            all_predictions.append(prediction.cpu().numpy())
            all_actual.append(Y.cpu().numpy())
            all_masks.append(mask.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    actual = np.concatenate(all_actual, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    print("\n--- R² Analysis (masked: land + no-seafloor-at-depth pixels excluded) ---")

    r2_scores = []

    # Calculate R² for each subsurface layer, ocean pixels only
    for layer in range(15):

        m = masks[:, layer, :, :].astype(bool).flatten()
        y_true = actual[:, layer, :, :].flatten()[m]
        y_pred = predictions[:, layer, :, :].flatten()[m]

        if y_true.size < 2:
            print(f"Layer {layer + 1:2d}: not enough valid ocean pixels in val set, skipping")
            r2_scores.append(float("nan"))
            continue

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

        r2_scores.append(r2)

        print(f"Layer {layer + 1:2d}: R² = {r2:.6f}  (n_valid_pixels={y_true.size})")

    # Find best and worst layers
    best_layer = int(np.nanargmax(r2_scores)) + 1
    worst_layer = int(np.nanargmin(r2_scores)) + 1

    print("\n--- Results ---")
    print(f"Best layer based on R²: Layer {best_layer}")
    print(f"Worst layer based on R²: Layer {worst_layer}")

    print("\nR² analysis completed successfully!")


if __name__ == "__main__":
    main()