"""
Training script for Audio Deepfake Detection Model.

Dataset structure required:
    dataset/audio/
        real/    ← .wav files of real human speech
        ai/     ← .wav files of AI-generated speech (ElevenLabs, Bark, etc.)

How it works:
    1. Reads .wav audio files
    2. Converts each to a Mel-spectrogram (visual representation of frequency over time)
    3. Treats spectrograms as grayscale images
    4. Trains a CNN (AudioCNN) to classify: Real (0) vs AI (1)

Usage:
    python train_audio.py
"""

import os
import sys
import wave
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from models.audio_cnn import AudioCNN

# ==========================================================
# CONFIG
# ==========================================================
DATASET_PATH = "dataset/audio"
MODEL_SAVE_PATH = "audio_model.pth"
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Spectrogram settings
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
SPEC_SIZE = 128  # Output spectrogram image size


# ==========================================================
# SPECTROGRAM GENERATION (No librosa needed)
# ==========================================================
def read_wav(path):
    """Read a WAV file and return audio as float32 numpy array."""
    try:
        with wave.open(path, 'rb') as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            audio_bytes = wf.readframes(n_frames)

            if wf.getsampwidth() == 2:
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif wf.getsampwidth() == 4:
                audio = np.frombuffer(audio_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                audio = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

            # Resample to target sample rate if needed
            if sr != SAMPLE_RATE and sr > 0:
                ratio = SAMPLE_RATE / sr
                new_len = int(len(audio) * ratio)
                audio = np.interp(np.linspace(0, len(audio)-1, new_len), np.arange(len(audio)), audio)

            return audio
    except Exception as e:
        print(f"   ⚠️ Could not read {path}: {e}")
        return None


def create_mel_spectrogram(audio):
    """
    Create a Mel-spectrogram from audio using pure numpy.
    Returns a 2D numpy array (frequency x time).
    """
    if audio is None or len(audio) < N_FFT:
        return None

    # Pad or truncate to fixed length (3 seconds)
    target_len = SAMPLE_RATE * 3
    if len(audio) > target_len:
        # Take middle section
        start = (len(audio) - target_len) // 2
        audio = audio[start:start + target_len]
    elif len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))

    # STFT
    num_frames = 1 + (len(audio) - N_FFT) // HOP_LENGTH
    spectrogram = np.zeros((N_FFT // 2 + 1, num_frames))

    window = np.hanning(N_FFT)

    for i in range(num_frames):
        start = i * HOP_LENGTH
        frame = audio[start:start + N_FFT] * window
        fft_result = np.fft.rfft(frame)
        spectrogram[:, i] = np.abs(fft_result)

    # Convert to Mel scale using triangular filter bank
    mel_filters = _create_mel_filterbank(N_FFT // 2 + 1, N_MELS, SAMPLE_RATE)
    mel_spec = mel_filters @ spectrogram

    # Log scale
    mel_spec = np.log1p(mel_spec)

    # Resize to fixed size
    from PIL import Image
    mel_img = Image.fromarray(mel_spec.astype(np.float32))
    mel_img = mel_img.resize((SPEC_SIZE, SPEC_SIZE))
    mel_spec = np.array(mel_img)

    # Normalize to [0, 1]
    if mel_spec.max() > mel_spec.min():
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())

    return mel_spec


def _create_mel_filterbank(n_fft_bins, n_mels, sample_rate):
    """Create triangular Mel filterbank matrix."""
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    mel_low = hz_to_mel(0)
    mel_high = hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((n_fft_bins - 1) * 2 * hz_points / sample_rate).astype(int)

    filters = np.zeros((n_mels, n_fft_bins))
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        for j in range(left, center):
            if center > left:
                filters[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                filters[i, j] = (right - j) / (right - center)

    return filters


# ==========================================================
# DATASET
# ==========================================================
class AudioDataset(Dataset):
    """Dataset that loads .wav files and converts them to spectrograms."""

    def __init__(self, root_dir, split="train", split_ratio=0.8):
        self.data = []

        real_dir = os.path.join(root_dir, "real")
        ai_dir = os.path.join(root_dir, "ai")

        real_files = [
            os.path.join(real_dir, f) for f in os.listdir(real_dir)
            if f.lower().endswith(('.wav', '.mp3', '.flac'))
        ] if os.path.exists(real_dir) else []

        ai_files = [
            os.path.join(ai_dir, f) for f in os.listdir(ai_dir)
            if f.lower().endswith(('.wav', '.mp3', '.flac'))
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

        print(f"   Audio {split}: {len(self.data)} samples "
              f"({sum(1 for _, l in self.data if l == 0)} real, "
              f"{sum(1 for _, l in self.data if l == 1)} AI)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]

        audio = read_wav(path)
        spectrogram = create_mel_spectrogram(audio)

        if spectrogram is None:
            # Return zeros for failed files
            spectrogram = np.zeros((SPEC_SIZE, SPEC_SIZE), dtype=np.float32)

        # Convert to tensor: (1, H, W)
        tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
        return tensor, label


# ==========================================================
# TRAINING
# ==========================================================
def train():
    print("=" * 60)
    print("TRAINING: Audio Deepfake Detection (AudioCNN)")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        print(f"\n❌ Dataset not found at: {DATASET_PATH}")
        print(f"   Create the following structure:")
        print(f"   {DATASET_PATH}/real/   ← .wav files of real speech")
        print(f"   {DATASET_PATH}/ai/    ← .wav files of AI-generated speech")
        return

    train_dataset = AudioDataset(DATASET_PATH, split="train")
    val_dataset = AudioDataset(DATASET_PATH, split="val")

    if len(train_dataset) == 0:
        print("❌ No training samples found!")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = AudioCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"\n   Device: {DEVICE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Learning Rate: {LR}")

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for spectrograms, labels in train_loader:
            spectrograms = spectrograms.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(spectrograms)
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
            for spectrograms, labels in val_loader:
                spectrograms = spectrograms.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(spectrograms)
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
