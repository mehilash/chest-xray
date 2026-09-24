import torch

MODEL_PATH = "best_chest_xray_model (2).pth"

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

print("=" * 50)
print("CHECKPOINT INFORMATION")
print("=" * 50)

print("\nEpoch:")
print(checkpoint["epoch"])

print("\nValidation F1:")
print(checkpoint["val_f1"])

print("\nClass names:")
print(checkpoint["class_names"])

print("\nTraining configuration:")
print(checkpoint["config"])

print("\n" + "=" * 50)