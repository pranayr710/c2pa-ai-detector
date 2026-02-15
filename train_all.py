import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset_loader import ImageDataset
from models.cnn3 import CNN3
from models.cnn6 import CNN6
from models.efficientnet import EfficientNetB0
from models.hybrid_fft import HybridFFT


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
BATCH_SIZE = 16
LR = 0.0001


def train_model(model, model_name):
    print("\n" + "=" * 60)
    print(f"TRAINING MODEL: {model_name}")
    print("=" * 60)

    train_dataset = ImageDataset("dataset", split="train")
    val_dataset = ImageDataset("dataset", split="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss:.3f}, Val Accuracy={acc:.2f}%")

    torch.save(model.state_dict(), f"{model_name}.pth")
    print(f"✅ Saved {model_name}.pth")


def main():
    models = {
        "cnn3": CNN3(),
        "cnn6": CNN6(),
        "efficientnet": EfficientNetB0(),
        "hybrid": HybridFFT(),
    }

    for name, model in models.items():
        train_model(model, name)


if __name__ == "__main__":
    main()
