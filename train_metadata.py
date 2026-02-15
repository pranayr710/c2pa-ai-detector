"""
Training script for Metadata Forensics ML Model.

Uses the EXISTING dataset (dataset/real/ and dataset/ai/) — no new dataset needed!

How it works:
    1. Scans images from dataset/real/ and dataset/ai/
    2. Extracts 15 metadata features from each image (EXIF, file info, dimensions)
    3. Trains a small MLP (MetadataMLP) to classify based on metadata patterns
    4. AI-generated images often have distinct metadata signatures
       (missing camera info, AI software markers, specific dimensions, etc.)

Usage:
    python train_metadata.py
"""

import os
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from PIL.ExifTags import TAGS

from models.metadata_mlp import MetadataMLP

# ==========================================================
# CONFIG
# ==========================================================
DATASET_PATH = "dataset"  # Uses existing dataset/real/ and dataset/ai/
MODEL_SAVE_PATH = "metadata_model.pth"
BATCH_SIZE = 32
EPOCHS = 15
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Known AI software signatures for feature extraction
AI_SOFTWARE = [
    "adobe firefly", "midjourney", "dall-e", "stable diffusion",
    "dreamstudio", "playground", "leonardo", "canva ai",
    "comfyui", "automatic1111", "invokeai", "fooocus",
    "runway", "pika", "nightcafe", "wombo", "craiyon"
]

AI_METADATA_KEYS = [
    "prompt", "negative_prompt", "cfg_scale", "sampler",
    "sd_model", "model_hash", "steps", "seed", "ai_generated"
]


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================
def extract_metadata_features(image_path):
    """
    Extract 15 numeric features from image metadata.
    Returns a numpy array of shape (15,).
    """
    features = np.zeros(15, dtype=np.float32)

    try:
        img = Image.open(image_path)

        # File-level features
        stat = os.stat(image_path)
        w, h = img.size

        # Feature 0: has_camera_make
        # Feature 1: has_camera_model
        # Feature 2: has_gps
        # Feature 3: has_datetime
        # Feature 4: has_datetime_original
        # Feature 5: has_software
        # Feature 6: has_ai_software
        # Feature 7: has_ai_metadata_key
        # Feature 8: exif_tag_count (normalized)
        # Feature 9: file_size_kb (normalized)
        # Feature 10: is_square
        # Feature 11: is_power_of_2
        # Feature 12: width_normalized
        # Feature 13: height_normalized
        # Feature 14: has_lens_info

        exif_data = None
        try:
            exif_data = img._getexif()
        except:
            pass

        all_tags = {}
        if exif_data:
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                try:
                    all_tags[tag_name] = str(value).lower()
                except:
                    all_tags[tag_name] = ""

        # Feature 0: has_camera_make
        features[0] = 1.0 if "Make" in all_tags else 0.0

        # Feature 1: has_camera_model
        features[1] = 1.0 if "Model" in all_tags else 0.0

        # Feature 2: has_gps
        features[2] = 1.0 if "GPSInfo" in all_tags else 0.0

        # Feature 3: has_datetime
        features[3] = 1.0 if "DateTime" in all_tags else 0.0

        # Feature 4: has_datetime_original
        features[4] = 1.0 if "DateTimeOriginal" in all_tags else 0.0

        # Feature 5: has_software
        features[5] = 1.0 if "Software" in all_tags else 0.0

        # Feature 6: has_ai_software
        all_values = " ".join(all_tags.values())
        features[6] = 1.0 if any(sig in all_values for sig in AI_SOFTWARE) else 0.0

        # Feature 7: has_ai_metadata_key
        all_keys_lower = [k.lower() for k in all_tags.keys()]
        # Also check PNG text chunks
        png_texts = {}
        if hasattr(img, "text") and img.text:
            png_texts = {k.lower(): v.lower() for k, v in img.text.items()}
            all_keys_lower.extend(png_texts.keys())
            all_values += " " + " ".join(png_texts.values())

        features[7] = 1.0 if any(key in " ".join(all_keys_lower) for key in AI_METADATA_KEYS) else 0.0

        # Feature 8: exif_tag_count (normalized, max ~50 tags)
        features[8] = min(len(all_tags) / 50.0, 1.0)

        # Feature 9: file_size_kb (normalized, max ~10MB)
        features[9] = min(stat.st_size / (10 * 1024 * 1024), 1.0)

        # Feature 10: is_square
        features[10] = 1.0 if w == h else 0.0

        # Feature 11: is_power_of_2
        power_of_2 = [256, 512, 768, 1024, 2048, 4096]
        features[11] = 1.0 if (w in power_of_2 and h in power_of_2) else 0.0

        # Feature 12: width_normalized (max 4096)
        features[12] = min(w / 4096.0, 1.0)

        # Feature 13: height_normalized (max 4096)
        features[13] = min(h / 4096.0, 1.0)

        # Feature 14: has_lens_info
        features[14] = 1.0 if ("LensModel" in all_tags or "LensMake" in all_tags) else 0.0

    except Exception as e:
        pass  # Return zeros for failed files

    return features


# ==========================================================
# DATASET
# ==========================================================
class MetadataDataset(Dataset):
    """Dataset that extracts metadata features from images."""

    def __init__(self, root_dir, split="train", split_ratio=0.8):
        self.data = []

        real_dir = os.path.join(root_dir, "real")
        ai_dir = os.path.join(root_dir, "ai")

        real_files = [
            os.path.join(real_dir, f) for f in os.listdir(real_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        ] if os.path.exists(real_dir) else []

        ai_files = [
            os.path.join(ai_dir, f) for f in os.listdir(ai_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        ] if os.path.exists(ai_dir) else []

        random.shuffle(real_files)
        random.shuffle(ai_files)

        real_split = int(len(real_files) * split_ratio)
        ai_split = int(len(ai_files) * split_ratio)

        if split == "train":
            self.data = (
                [(p, 0) for p in real_files[:real_split]] +
                [(p, 1) for p in ai_files[:ai_split]]
            )
        else:
            self.data = (
                [(p, 0) for p in real_files[real_split:]] +
                [(p, 1) for p in ai_files[ai_split:]]
            )

        print(f"   Metadata {split}: {len(self.data)} samples "
              f"({sum(1 for _, l in self.data if l == 0)} real, "
              f"{sum(1 for _, l in self.data if l == 1)} AI)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        features = extract_metadata_features(img_path)
        return torch.tensor(features, dtype=torch.float32), label


# ==========================================================
# TRAINING
# ==========================================================
def train():
    print("=" * 60)
    print("TRAINING: Metadata Forensics (MetadataMLP)")
    print("=" * 60)

    if not os.path.exists(os.path.join(DATASET_PATH, "real")):
        print(f"\n❌ Dataset not found at: {DATASET_PATH}")
        print(f"   Expected: {DATASET_PATH}/real/ and {DATASET_PATH}/ai/")
        return

    train_dataset = MetadataDataset(DATASET_PATH, split="train")
    val_dataset = MetadataDataset(DATASET_PATH, split="val")

    if len(train_dataset) == 0:
        print("❌ No training samples found!")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = MetadataMLP(input_size=15).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\n   Device: {DEVICE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Features: 15 metadata features per image")

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for features, labels in train_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(features)
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
            for features, labels in val_loader:
                features = features.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(features)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total if total > 0 else 0
        print(f"   Epoch {epoch+1}/{EPOCHS}: Loss={total_loss:.3f}, Val Accuracy={acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ Model saved: {MODEL_SAVE_PATH}")
    print(f"   Best Validation Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    train()
