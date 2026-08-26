import torch
import numpy as np
from model import OceanSubsurfaceAutoencoder
from data_loader import load_data


def main():

    # Select device
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

    # Store metrics
    mse_scores = []
    mae_scores = []
    r2_scores = []

    print("\n========== COMBINED LAYER ANALYSIS ==========")
    print("\nLayer       MSE          MAE          R²")
    print("----------------------------------------------")

    # Calculate metrics for each layer
    for layer in range(15):

        y_true = actual[:, layer, :, :].flatten()
        y_pred = predictions[:, layer, :, :].flatten()

        # MSE
        mse = np.mean((y_true - y_pred) ** 2)

        # MAE
        mae = np.mean(np.abs(y_true - y_pred))

        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        r2 = 1 - (ss_res / ss_tot)

        mse_scores.append(mse)
        mae_scores.append(mae)
        r2_scores.append(r2)

        print(
            f"{layer + 1:5d}   "
            f"{mse:10.6f}   "
            f"{mae:10.6f}   "
            f"{r2:10.6f}"
        )

    # Find best and worst layers
    best_mse_layer = np.argmin(mse_scores) + 1
    worst_mse_layer = np.argmax(mse_scores) + 1

    best_mae_layer = np.argmin(mae_scores) + 1
    worst_mae_layer = np.argmax(mae_scores) + 1

    best_r2_layer = np.argmax(r2_scores) + 1
    worst_r2_layer = np.argmin(r2_scores) + 1

    print("\n========== RESULTS ==========")

    print(f"Best layer based on MSE: Layer {best_mse_layer}")
    print(f"Worst layer based on MSE: Layer {worst_mse_layer}")

    print(f"Best layer based on MAE: Layer {best_mae_layer}")
    print(f"Worst layer based on MAE: Layer {worst_mae_layer}")

    print(f"Best layer based on R²: Layer {best_r2_layer}")
    print(f"Worst layer based on R²: Layer {worst_r2_layer}")

    print("\nCombined analysis completed successfully!")


if __name__ == "__main__":
    main()