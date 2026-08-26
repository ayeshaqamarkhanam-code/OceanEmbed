import torch
import torch.nn as nn

from model import OceanSubsurfaceAutoencoder
from data_loader import load_data


def main():

    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    # Load validation data
    train_loader, val_loader = load_data()

    # Create model
    model = OceanSubsurfaceAutoencoder(
        in_channels=5,
        out_channels=15
    )

    # Load the trained model
    model.load_state_dict(
        torch.load(
            "best_ocean_model.pth",
            map_location=device,
            weights_only=True
        )
    )

    model.to(device)
    model.eval()

    # Loss function
    criterion = nn.MSELoss()

    total_loss = 0.0
    total_mae = 0.0
    batches = 0

    print("\n--- Model Evaluation ---")

    with torch.no_grad():

        for X, Y in val_loader:

            X = X.to(device)
            Y = Y.to(device)

            # Prediction
            prediction = model(X)

            # MSE
            loss = criterion(prediction, Y)

            # MAE
            mae = torch.mean(torch.abs(prediction - Y))

            total_loss += loss.item()
            total_mae += mae.item()

            batches += 1

    average_loss = total_loss / batches
    average_mae = total_mae / batches

    print("Validation MSE:", average_loss)
    print("Validation MAE:", average_mae)

    print("\nModel evaluation completed successfully!")


if __name__ == "__main__":
    main()