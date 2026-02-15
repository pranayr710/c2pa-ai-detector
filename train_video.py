"""
Training script for Video AI Detection Model.

This trains a CNN specifically on VIDEO FRAMES, not clean photographs.
This is critical because video compression artifacts confuse image models.

Dataset structure:
    dataset/video/
        real/   ← .mp4/.avi files of REAL videos (camera-recorded, screen recordings, etc.)
        ai/     ← .mp4/.avi files of AI-GENERATED videos (Sora, Runway, Pika, etc.)

How it works:
    1. Extracts frames from each video (every Nth frame)
    2. The frames INCLUDE compression artifacts — this is intentional!
    3. Trains VideoCNN to tell the difference between:
       - Real video frames (WITH compression) = class 0
       - AI video frames (WITH compression) = class 1

Usage:
    python train_video.py
"""

import os
import random
import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from models.video_cnn import VideoCNN

# ==========================================================
# CONFIG
# ==========================================================
DATASET_PATH = "vedio_database"
MODEL_SAVE_PATH = "video_model.pth"
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# How many frames to extract per video
FRAMES_PER_VIDEO = 30

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
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================================
# FRAME EXTRACTION
# ==========================================================
def extract_frames_from_video(video_path, num_frames=FRAMES_PER_VIDEO):
    """
    Extract evenly-spaced frames from a video file.
    Returns a list of PIL Images.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # Pick evenly spaced frame indices
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        step = total_frames / num_frames
        indices = [int(i * step) for i in range(num_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            frames.append(pil_img)

    cap.release()
    return frames


# ==========================================================
# DATASET
# ==========================================================
class VideoFrameDataset(Dataset):
    """
    Loads video files, extracts frames, and serves them as training samples.
    Each frame inherits its video's label (real=0, ai=1).
    """

    def __init__(self, root_dir, split="train", split_ratio=0.8, transform=None):
        self.transform = transform or TRANSFORM
        self.frames = []  # List of (PIL Image, label)

        real_dir = os.path.join(root_dir, "real")
        ai_dir = os.path.join(root_dir, "ai")

        video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv')

        # Collect video files
        real_videos = [
            os.path.join(real_dir, f) for f in os.listdir(real_dir)
            if f.lower().endswith(video_exts)
        ] if os.path.exists(real_dir) else []

        ai_videos = [
            os.path.join(ai_dir, f) for f in os.listdir(ai_dir)
            if f.lower().endswith(video_exts)
        ] if os.path.exists(ai_dir) else []

        # Split videos (not frames) for proper train/val separation
        random.shuffle(real_videos)
        random.shuffle(ai_videos)

        real_split = int(len(real_videos) * split_ratio)
        ai_split = int(len(ai_videos) * split_ratio)

        if split == "train":
            selected_real = real_videos[:real_split]
            selected_ai = ai_videos[:ai_split]
        else:
            selected_real = real_videos[real_split:]
            selected_ai = ai_videos[ai_split:]

        # Extract frames from selected videos
        print(f"   Extracting {split} frames...")
        for video_path in selected_real:
            frames = extract_frames_from_video(video_path)
            for f in frames:
                self.frames.append((f, 0))  # label 0 = REAL
            if frames:
                print(f"      ✅ {os.path.basename(video_path)}: {len(frames)} frames (REAL)")

        for video_path in selected_ai:
            frames = extract_frames_from_video(video_path)
            for f in frames:
                self.frames.append((f, 1))  # label 1 = AI
            if frames:
                print(f"      ✅ {os.path.basename(video_path)}: {len(frames)} frames (AI)")

        random.shuffle(self.frames)

        real_count = sum(1 for _, l in self.frames if l == 0)
        ai_count = sum(1 for _, l in self.frames if l == 1)
        print(f"   Video {split}: {len(self.frames)} frames ({real_count} real, {ai_count} AI) "
              f"from {len(selected_real) + len(selected_ai)} videos")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        pil_img, label = self.frames[idx]
        tensor = self.transform(pil_img)
        return tensor, label


# ==========================================================
# TRAINING
# ==========================================================
def train():
    print("=" * 60)
    print("TRAINING: Video AI Detection (VideoCNN)")
    print("=" * 60)
    print("This model is trained ON video frames (with compression)")
    print("so it won't confuse compression artifacts with AI.\n")

    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at: {DATASET_PATH}")
        print(f"\n   Create the following structure:")
        print(f"   {DATASET_PATH}/real/   ← .mp4 files of REAL videos")
        print(f"   {DATASET_PATH}/ai/    ← .mp4 files of AI-GENERATED videos")
        print(f"\n   Examples of AI video sources:")
        print(f"   - Sora (OpenAI)")
        print(f"   - Runway Gen-2/Gen-3")
        print(f"   - Pika Labs")
        print(f"   - Stable Video Diffusion")
        print(f"   - Kling AI")
        return

    print("📦 Loading video datasets...\n")
    train_dataset = VideoFrameDataset(DATASET_PATH, split="train", transform=TRAIN_TRANSFORM)
    val_dataset = VideoFrameDataset(DATASET_PATH, split="val", transform=TRANSFORM)

    if len(train_dataset) == 0:
        print("❌ No training frames extracted!")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = VideoCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\n   Device: {DEVICE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Train frames: {len(train_dataset)}")
    print(f"   Val frames: {len(val_dataset)}")

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

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
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
