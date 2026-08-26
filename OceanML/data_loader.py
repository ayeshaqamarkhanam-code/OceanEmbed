import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class OceanDataset(Dataset):
    def __init__(self, x_data, y_data, mean_x, std_x, mean_y, std_y):
        # Convert NumPy arrays to PyTorch tensors
        self.x = torch.tensor(x_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32)

        # Store normalization values
        self.mean_x = mean_x
        self.std_x = std_x
        self.mean_y = mean_y
        self.std_y = std_y

        # Normalize X and Y
        self.x = (self.x - self.mean_x) / self.std_x
        self.y = (self.y - self.mean_y) / self.std_y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_data(
    x_path="INCOIS_X_surface.npy",
    y_path="INCOIS_Y_subsurface.npy",
    batch_size=2
):
    # -------------------------------------------------
    # 1. Load the NumPy files
    # -------------------------------------------------
    x_data = np.load(x_path)
    y_data = np.load(y_path)

    print("Original X shape:", x_data.shape)
    print("Original Y shape:", y_data.shape)

    # -------------------------------------------------
    # 2. Check that X and Y match
    # -------------------------------------------------
    if x_data.shape[0] != y_data.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")

    if x_data.shape[0] != 10:
        raise ValueError(
            f"Expected 10 samples, but found {x_data.shape[0]}."
        )

    if x_data.shape[1:] != (5, 101, 241):
        raise ValueError(
            f"Unexpected X shape: {x_data.shape}"
        )

    if y_data.shape[1:] != (15, 101, 241):
        raise ValueError(
            f"Unexpected Y shape: {y_data.shape}"
        )

    # -------------------------------------------------
    # 3. Split into training and validation data
    #    8 samples = training
    #    2 samples = validation
    # -------------------------------------------------
    x_train = x_data[:8]
    y_train = y_data[:8]

    x_val = x_data[8:]
    y_val = y_data[8:]

    print("Training X shape:", x_train.shape)
    print("Training Y shape:", y_train.shape)
    print("Validation X shape:", x_val.shape)
    print("Validation Y shape:", y_val.shape)

    # -------------------------------------------------
    # 4. Calculate normalization statistics
    #    ONLY from training data
    # -------------------------------------------------
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    mean_x = x_train_tensor.mean(dim=(0, 2, 3), keepdim=True)
    std_x = x_train_tensor.std(
        dim=(0, 2, 3),
        keepdim=True,
        unbiased=False
    )

    mean_y = y_train_tensor.mean(dim=(0, 2, 3), keepdim=True)
    std_y = y_train_tensor.std(
        dim=(0, 2, 3),
        keepdim=True,
        unbiased=False
    )

    # Prevent division by zero
    std_x = torch.clamp(std_x, min=1e-8)
    std_y = torch.clamp(std_y, min=1e-8)

    # -------------------------------------------------
    # 5. Create Dataset objects
    # -------------------------------------------------
    train_dataset = OceanDataset(
        x_train,
        y_train,
        mean_x,
        std_x,
        mean_y,
        std_y
    )

    val_dataset = OceanDataset(
        x_val,
        y_val,
        mean_x,
        std_x,
        mean_y,
        std_y
    )

    # -------------------------------------------------
    # 6. Create PyTorch DataLoaders
    # -------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader


# -----------------------------------------------------
# 7. Test the DataLoader
# -----------------------------------------------------
if __name__ == "__main__":

    train_loader, val_loader = load_data()

    print("\n--- DataLoader Verification ---")

    print("Number of training batches:", len(train_loader))
    print("Number of validation batches:", len(val_loader))

    # Get one batch
    x_batch, y_batch = next(iter(train_loader))

    print("X batch shape:", x_batch.shape)
    print("Y batch shape:", y_batch.shape)

    print("\nData Loader is working successfully!")