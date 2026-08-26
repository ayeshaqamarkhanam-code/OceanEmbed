import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import load_data
from model import OceanSubsurfaceAutoencoder


def main():

    # --------------------------------------------------
    # 1. Select device
    # --------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("========================================")
    print("OceanML Training")
    print("========================================")
    print("Using device:", device)

    # --------------------------------------------------
    # 2. Load training and validation data
    # --------------------------------------------------
    train_loader, val_loader = load_data()

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    # --------------------------------------------------
    # 3. Create model
    # --------------------------------------------------
    model = OceanSubsurfaceAutoencoder(
        in_channels=5,
        out_channels=15
    )

    model = model.to(device)

    # --------------------------------------------------
    # 4. Loss function
    # --------------------------------------------------
    criterion = nn.MSELoss()

    # --------------------------------------------------
    # 5. Optimizer
    # --------------------------------------------------
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # --------------------------------------------------
    # 6. Training settings
    # --------------------------------------------------
    num_epochs = 20

    best_val_loss = float("inf")

    # --------------------------------------------------
    # 7. Training loop
    # --------------------------------------------------
    for epoch in range(num_epochs):

        model.train()

        total_train_loss = 0.0

        for X, Y in train_loader:

            X = X.to(device)
            Y = Y.to(device)

            # Clear old gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(X)

            # Calculate error
            loss = criterion(predictions, Y)

            # Backpropagation
            loss.backward()

            # Update model weights
            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = (
            total_train_loss / len(train_loader)
        )

        # --------------------------------------------------
        # 8. Validation
        # --------------------------------------------------
        model.eval()

        total_val_loss = 0.0

        with torch.no_grad():

            for X, Y in val_loader:

                X = X.to(device)
                Y = Y.to(device)

                predictions = model(X)

                loss = criterion(predictions, Y)

                total_val_loss += loss.item()

        average_val_loss = (
            total_val_loss / len(val_loader)
        )

        # --------------------------------------------------
        # 9. Print progress
        # --------------------------------------------------
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {average_train_loss:.6f} "
            f"Val Loss: {average_val_loss:.6f}"
        )

        # --------------------------------------------------
        # 10. Save best model
        # --------------------------------------------------
        if average_val_loss < best_val_loss:

            best_val_loss = average_val_loss

            torch.save(
                model.state_dict(),
                "best_ocean_model.pth"
            )

            print("  -> Best model saved!")

    print()
    print("========================================")
    print("Training completed!")
    print("Best validation loss:", best_val_loss)
    print("Saved model: best_ocean_model.pth")
    print("========================================")


if __name__ == "__main__":
    main()