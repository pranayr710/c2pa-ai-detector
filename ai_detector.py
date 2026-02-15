import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image

# ==========================================================
# CONFIG
# ==========================================================
DATASET_PATH = r"C:\Users\Vasala Vignesh\Documents\pranay\final\dataset"
MODEL_PATH = "ai_detector_model.pth"
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# TRANSFORMS
# ==========================================================
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==========================================================
# MODEL
# ==========================================================
def build_model():
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model

# ==========================================================
# TRAINING FUNCTION (RUN ONCE)
# ==========================================================
def train_model():
    dataset = ImageFolder(DATASET_PATH, transform=train_transform)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = build_model().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("🚀 Training started...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("✅ Model saved:", MODEL_PATH)

# ==========================================================
# LOAD MODEL (FOR INFERENCE)
# ==========================================================
_model = None

def load_model():
    global _model
    if _model is None:
        model = build_model().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        _model = model
    return _model

# ==========================================================
# AI DETECTION FUNCTION (USED BY verifier.py)
# ==========================================================
def ai_detect(image_path: str) -> float:
    """
    Returns:
        score between 0 and 1
        (closer to 1 = AI generated)
    """
    model = load_model()

    img = Image.open(image_path).convert("RGB")
    img = test_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img)
        score = torch.sigmoid(output).item()

    return score


# ==========================================================
# TRAIN ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        train_model()
    else:
        print("Model already trained.")
