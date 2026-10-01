from pathlib import Path
import argparse, json, time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def train(data_dir, output, epochs=8, batch_size=32, lr=3e-4, image_size=224):
    data_dir = Path(data_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.15),
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
    ])

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)

    print("Classes:", len(train_ds.classes))
    print("Train:", len(train_ds), "Val:", len(val_ds), "Test:", len(test_ds))

    workers = 0 if device.type == "cpu" else 2
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers)

    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_ds.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = -1.0
    best_state = None

    def evaluate(loader):
        model.eval()
        correct = total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss_sum += criterion(logits, y).item() * y.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        return loss_sum / max(total,1), correct / max(total,1)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        start = time.time()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)
        scheduler.step()
        val_loss, val_acc = evaluate(val_dl)
        print(f"Epoch {epoch}/{epochs} train_loss={running/max(seen,1):.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={time.time()-start:.1f}s")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(test_dl)
    print(f"TEST accuracy={test_acc:.4f} loss={test_loss:.4f}")

    torch.save({
        "model_name": "efficientnet_b0",
        "state_dict": model.state_dict(),
        "classes": train_ds.classes,
        "image_size": image_size,
        "mean": [.485,.456,.406],
        "std": [.229,.224,.225],
    }, output)

    print("Saved:", output)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()
    train(args.data, args.output, args.epochs, args.batch_size, args.lr)
