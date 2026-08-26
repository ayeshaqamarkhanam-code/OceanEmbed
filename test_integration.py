from data_loader import get_dataloaders
from model import OceanSubsurfaceAutoencoder

# 1. Person A's pipeline creates batches
train_loader, val_loader, mean_y, std_y = get_dataloaders()
sample_x, sample_y = next(iter(train_loader))

# 2. Person B's model processes the batch
model = OceanSubsurfaceAutoencoder()
predictions = model(sample_x)

# 3. Verify output tensor shape matches target
print("\n--- Integration Verification ---")
print("Input Batch Shape: ", sample_x.shape)        # Expected: torch.Size([2, 5, 101, 241])
print("Target Batch Shape:", sample_y.shape)        # Expected: torch.Size([2, 15, 101, 241])
print("Output Batch Shape:", predictions.shape)     # Expected: torch.Size([2, 15, 101, 241])
print("--> Steps 1 & 2 integration successfully verified!\n")