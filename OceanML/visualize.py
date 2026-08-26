import torch
import matplotlib.pyplot as plt

from model import OceanSubsurfaceAutoencoder
from data_loader import load_data


def main():

    # Use GPU if available
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

    # Get one validation batch
    X, Y = next(iter(val_loader))

    X = X.to(device)
    Y = Y.to(device)

    # Make prediction
    with torch.no_grad():
        prediction = model(X)

    # Take first sample and first subsurface layer
    actual = Y[0, 0].cpu().numpy()
    predicted = prediction[0, 0].cpu().numpy()

    # Calculate absolute error
    error = abs(actual - predicted)

    print("\n--- Visualization ---")
    print("Actual shape:", actual.shape)
    print("Predicted shape:", predicted.shape)
    print("Mean absolute error:", error.mean())

    # Create three plots
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(actual, aspect="auto")
    plt.title("Actual Subsurface")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(predicted, aspect="auto")
    plt.title("Predicted Subsurface")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(error, aspect="auto")
    plt.title("Absolute Error")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar()

    plt.tight_layout()

    # Save image
    plt.savefig("prediction_visualization.png", dpi=150)

    print("\nVisualization saved as: prediction_visualization.png")

    # Show image
    plt.show()


if __name__ == "__main__":
    main()