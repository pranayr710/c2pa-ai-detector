
import dataset_loader
print("USING DATASET LOADER FROM:", dataset_loader.__file__)


import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset_loader import ImageDataset
from models.cnn3 import CNN3
from models.cnn6 import CNN6
from models.efficientnet import EfficientNetB0
from models.hybrid_fft import HybridFFT



def get_model(name):
    if name == "cnn3":
        return CNN3()
    elif name == "cnn6":
        return CNN6()
    elif name == "efficientnet":
        return EfficientNetB0()
    elif name == "hybrid":
        return HybridFFT()
    else:
        raise ValueError("Unknown model name")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    required=True,
    choices=["cnn3", "cnn6", "efficientnet", "hybrid"]
)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ✅ Dataset with split support
train_dataset = ImageDataset("dataset", split="train")
val_dataset = ImageDataset("dataset", split="val")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

model = get_model(args.model).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100.0 * correct / total
    print(f"Epoch {epoch+1}: Loss={total_loss:.3f}, Val Accuracy={acc:.2f}%")

torch.save(model.state_dict(), f"{args.model}.pth")
print(f"✅ Model saved as {args.model}.pth")
