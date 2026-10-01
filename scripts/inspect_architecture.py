import torch
from pathlib import Path

files = [
    Path("models/crop_classifier/crop_classifier_best.pth"),
    Path("models/disease/black_gram/best_model.pth"),
]

for file in files:
    print("\n" + "=" * 80)
    print(file)
    print("=" * 80)

    checkpoint = torch.load(
        file,
        map_location="cpu",
        weights_only=False
    )

    state_dict = checkpoint["model_state_dict"]

    print("\nNumber of parameters/tensors:", len(state_dict))

    print("\nModel state_dict keys and shapes:")

    for key, value in state_dict.items():
        print(f"{key:60s} {tuple(value.shape)}")