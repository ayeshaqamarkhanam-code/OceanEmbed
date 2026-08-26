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

    print("\n========== OVERALL MODEL METRICS ==========")

    # Flatten everything together
    y_true = actual.flatten()
    y_pred = predictions.flatten()

    # MSE
    mse = np.mean((y_true - y_pred) ** 2)

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1 - (ss_res / ss_tot)

    print(f"Overall MSE : {mse:.6f}")
    print(f"Overall MAE : {mae:.6f}")
    print(f"Overall R²  : {r2:.6f}")

    print("\n========== LAYER AVERAGES ==========")

    # Calculate each layer's metrics
    layer_mse = []
    layer_mae = []
    layer_r2 = []

    for layer in range(15):

        true_layer = actual[:, layer, :, :].flatten()
        pred_layer = predictions[:, layer, :, :].flatten()

        mse_layer = np.mean((true_layer - pred_layer) ** 2)
        mae_layer = np.mean(np.abs(true_layer - pred_layer))

        ss_res_layer = np.sum((true_layer - pred_layer) ** 2)
        ss_tot_layer = np.sum(
            (true_layer - np.mean(true_layer)) ** 2
        )

        r2_layer = 1 - (ss_res_layer / ss_tot_layer)

        layer_mse.append(mse_layer)
        layer_mae.append(mae_layer)
        layer_r2.append(r2_layer)

    print(f"Mean Layer MSE : {np.mean(layer_mse):.6f}")
    print(f"Mean Layer MAE : {np.mean(layer_mae):.6f}")
    print(f"Mean Layer R²  : {np.mean(layer_r2):.6f}")

    print("\nCombined metrics analysis completed successfully!")


if __name__ == "__main__":
    main()