import torch
from pathlib import Path

model_files = list(Path("models").rglob("*.pth"))

print(f"Found {len(model_files)} model files")

for file in model_files:
    print("\n" + "=" * 70)
    print(file)
    print("=" * 70)

    try:
        checkpoint = torch.load(
            file,
            map_location="cpu",
            weights_only=False
        )

        print("Type:", type(checkpoint))

        if isinstance(checkpoint, dict):
            print("Keys:")
            for key in checkpoint.keys():
                print("  -", key)

            # Show useful metadata without printing huge tensors
            for key, value in checkpoint.items():
                if key in ["classes", "class_names", "num_classes", "epoch",
                           "accuracy", "val_accuracy", "test_accuracy"]:
                    print(f"{key}: {value}")

        else:
            print("Checkpoint is not a dictionary.")

    except Exception as e:
        print("ERROR:", repr(e))