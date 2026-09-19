"""Alternative trainer: PyTorch ResNet18 transfer learning."""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

IMG_SIZE = 224


def loaders(data_dir: str, batch_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=train_tf)
    val_len = int(0.2 * len(dataset))
    train_set, val_set = random_split(dataset, [len(dataset) - val_len, val_len])
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4),
        DataLoader(val_set, batch_size=batch_size, num_workers=4),
        dataset.classes,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/PlantVillage")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="ml/models")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, val_dl, classes = loaders(args.data, args.batch_size)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    best_acc = 0.0
    os.makedirs(args.out, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_dl:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / max(total, 1)
        print(f"epoch {epoch + 1}/{args.epochs} val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.out, "crop_model.pt"))

    with open(os.path.join(args.out, "class_names.json"), "w", encoding="utf-8") as fh:
        json.dump(classes, fh, indent=2)


if __name__ == "__main__":
    main()
