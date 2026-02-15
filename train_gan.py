"""
Training script for GAN Fingerprint Identification Model.

Dataset structure required:
    dataset/generators/
        real/              ← real photographs
        midjourney/        ← Midjourney-generated images
        dalle/             ← DALL-E-generated images
        stable_diffusion/  ← Stable Diffusion-generated images

    Each subfolder name becomes a class label.
    You can add more generator folders as needed.

How it works:
    1. Loads images from each generator's folder
    2. Uses dual-stream CNN (spatial + FFT frequency analysis)
    3. Trains multi-class classifier
    4. Each generator leaves unique frequency fingerprints

Usage:
    python train_gan.py
"""

import os
import random
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from models.gan_classifier import GANClassifier

# ==========================================================
# CONFIG
# ==========================================================
DATASET_PATH = "dataset/generators"
MODEL_SAVE_PATH = "gan_model.pth"
CLASS_MAP_PATH = "gan_classes.txt"
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================================
# DATASET
# ==========================================================
class GANDataset(Dataset):
    """
    Multi-class dataset for generator identification.
    Each subfolder in root_dir is a class.
    """

    def __init__(self, root_dir, split="train", split_ratio=0.8, transform=None):
        self.data = []
        self.transform = transform or TRANSFORM
        self.classes = []

        if not os.path.exists(root_dir):
            print(f"⚠️ Dataset path not found: {root_dir}")
            return

        # Each subdirectory = one class
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            cls_idx = self.class_to_idx[cls_name]

            files = [
                os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
            ]

            random.shuffle(files)
            split_point = int(len(files) * split_ratio)

            if split == "train":
                selected = files[:split_point]
            else:
                selected = files[split_point:]

            self.data.extend([(f, cls_idx) for f in selected])

        random.shuffle(self.data)

        print(f"   GAN {split}: {len(self.data)} samples across {len(self.classes)} classes")
        for cls_name in self.classes:
            idx = self.class_to_idx[cls_name]
            count = sum(1 for _, l in self.data if l == idx)
            print(f"      {cls_name}: {count} images")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label


# ==========================================================
# TRAINING
# ==========================================================
def train():
    print("=" * 60)
    print("TRAINING: GAN Fingerprint Identification (GANClassifier)")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        print(f"\n❌ Dataset not found at: {DATASET_PATH}")
        print(f"   Create the following structure:")
        print(f"   {DATASET_PATH}/real/             ← real photos")
        print(f"   {DATASET_PATH}/midjourney/       ← Midjourney images")
        print(f"   {DATASET_PATH}/dalle/            ← DALL-E images")
        print(f"   {DATASET_PATH}/stable_diffusion/ ← Stable Diffusion images")
        print(f"   (Add more folders for more generators)")
        return

    train_dataset = GANDataset(DATASET_PATH, split="train", transform=TRAIN_TRANSFORM)
    val_dataset = GANDataset(DATASET_PATH, split="val", transform=TRANSFORM)

    if len(train_dataset) == 0:
        print("❌ No training samples found!")
        return

    num_classes = len(train_dataset.classes)
    print(f"\n   Detected {num_classes} classes: {train_dataset.classes}")

    # Save class mapping for inference
    with open(CLASS_MAP_PATH, "w") as f:
        for cls_name in train_dataset.classes:
            f.write(f"{cls_name}\n")
    print(f"   Class map saved to: {CLASS_MAP_PATH}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = GANClassifier(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\n   Device: {DEVICE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Classes: {num_classes}")

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

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
        class_correct = {}
        class_total = {}

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                preds = outputs.argmax(dim=1)

                for pred, label in zip(preds, labels):
                    total += 1
                    cls_name = train_dataset.classes[label.item()]
                    class_total[cls_name] = class_total.get(cls_name, 0) + 1
                    if pred == label:
                        correct += 1
                        class_correct[cls_name] = class_correct.get(cls_name, 0) + 1

        acc = 100 * correct / total if total > 0 else 0
        print(f"   Epoch {epoch+1}/{EPOCHS}: Loss={total_loss:.3f}, Val Accuracy={acc:.2f}%")

        # Per-class accuracy on last epoch
        if epoch == EPOCHS - 1:
            print("\n   Per-class accuracy:")
            for cls_name in train_dataset.classes:
                c = class_correct.get(cls_name, 0)
                t = class_total.get(cls_name, 0)
                cls_acc = 100 * c / t if t > 0 else 0
                print(f"      {cls_name:20s}: {cls_acc:.1f}% ({c}/{t})")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ Model saved: {MODEL_SAVE_PATH}")
    print(f"   Best Validation Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    train()
