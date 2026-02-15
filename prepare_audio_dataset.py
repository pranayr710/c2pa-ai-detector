"""
Prepare balanced audio dataset for training.

Reads from:
    dataset_voice/FAKE/   ← AI voice clones (.wav, long files)
    dataset_voice/REAL/   ← Real speech (.flac, many short clips in subfolders)

Outputs to:
    dataset/audio/ai/     ← 3-second WAV chunks from FAKE
    dataset/audio/real/   ← 3-second WAV chunks from REAL

Balancing strategy:
    1. Calculate total FAKE duration
    2. Pick real clips from different speakers until total real duration >= fake duration
    3. Chop everything into 3-second WAV chunks (matching the training pipeline)
"""

import os
import wave
import struct
import shutil
import random
import numpy as np

# ==========================================================
# CONFIG
# ==========================================================
SOURCE_FAKE = "dataset_voice/FAKE"
SOURCE_REAL = "dataset_voice/REAL"
OUTPUT_AI = "dataset/audio/ai"
OUTPUT_REAL = "dataset/audio/real"

CHUNK_DURATION = 3  # seconds — matches AudioCNN training
TARGET_SAMPLE_RATE = 16000


# ==========================================================
# AUDIO I/O
# ==========================================================
def read_audio_file(path):
    """Read .wav or .flac file as float32 numpy array + sample rate."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".flac":
        try:
            import soundfile as sf
            audio, sr = sf.read(path, dtype='float32')
            # Convert stereo to mono
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            return audio, sr
        except Exception as e:
            return None, None

    elif ext == ".wav":
        try:
            with wave.open(path, 'rb') as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                raw = wf.readframes(n_frames)

                if sample_width == 2:
                    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                elif sample_width == 4:
                    audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

                if n_channels > 1:
                    audio = audio.reshape(-1, n_channels)[:, 0]

                return audio, sr
        except Exception as e:
            return None, None

    return None, None


def save_wav_chunk(audio_data, sample_rate, output_path):
    """Save float32 audio array as 16-bit PCM WAV."""
    audio_int16 = (audio_data * 32767).astype(np.int16)

    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def resample(audio, original_sr, target_sr):
    """Simple linear resampling."""
    if original_sr == target_sr:
        return audio
    ratio = target_sr / original_sr
    new_len = int(len(audio) * ratio)
    return np.interp(np.linspace(0, len(audio) - 1, new_len), np.arange(len(audio)), audio)


# ==========================================================
# CHUNKING
# ==========================================================
def chop_into_chunks(audio, sr, chunk_sec=CHUNK_DURATION):
    """Split audio into fixed-length chunks. Drop the tail if too short."""
    chunk_samples = int(sr * chunk_sec)
    chunks = []
    for start in range(0, len(audio) - chunk_samples + 1, chunk_samples):
        chunk = audio[start:start + chunk_samples]
        chunks.append(chunk)
    return chunks


# ==========================================================
# MAIN
# ==========================================================
def prepare_dataset():
    print("=" * 60)
    print("PREPARING BALANCED AUDIO DATASET")
    print("=" * 60)

    # Create output directories
    os.makedirs(OUTPUT_AI, exist_ok=True)
    os.makedirs(OUTPUT_REAL, exist_ok=True)

    # ==========================================
    # STEP 1: Process FAKE audio
    # ==========================================
    print("\n📂 Processing FAKE (AI) audio...")
    fake_files = [
        os.path.join(SOURCE_FAKE, f) for f in os.listdir(SOURCE_FAKE)
        if f.lower().endswith(('.wav', '.mp3', '.flac'))
    ]

    total_fake_duration = 0.0
    ai_chunk_count = 0

    for i, path in enumerate(fake_files):
        print(f"   [{i+1}/{len(fake_files)}] {os.path.basename(path)}...", end="")
        audio, sr = read_audio_file(path)
        if audio is None:
            print(" SKIP")
            continue

        # Resample
        audio = resample(audio, sr, TARGET_SAMPLE_RATE)
        duration = len(audio) / TARGET_SAMPLE_RATE
        total_fake_duration += duration

        # Chop into 3-second chunks
        chunks = chop_into_chunks(audio, TARGET_SAMPLE_RATE)
        for j, chunk in enumerate(chunks):
            out_name = f"ai_{i:03d}_{j:04d}.wav"
            save_wav_chunk(chunk, TARGET_SAMPLE_RATE, os.path.join(OUTPUT_AI, out_name))
            ai_chunk_count += 1

        print(f" {duration:.0f}s → {len(chunks)} chunks")

    print(f"\n   ✅ FAKE total: {total_fake_duration:.0f}s ({total_fake_duration/60:.1f} min), {ai_chunk_count} chunks")

    # ==========================================
    # STEP 2: Collect REAL audio files
    # ==========================================
    print(f"\n📂 Collecting REAL audio files...")
    real_files = []
    for root, dirs, files in os.walk(SOURCE_REAL):
        for f in files:
            if f.lower().endswith(('.wav', '.flac', '.mp3')):
                real_files.append(os.path.join(root, f))

    print(f"   Found {len(real_files)} real audio files")

    # Shuffle to get variety from different speakers
    random.shuffle(real_files)

    # ==========================================
    # STEP 3: Pick real clips to match fake duration
    # ==========================================
    print(f"\n📂 Processing REAL audio (target: ~{total_fake_duration:.0f}s to match FAKE)...")

    total_real_duration = 0.0
    real_chunk_count = 0
    files_used = 0

    for i, path in enumerate(real_files):
        # Stop when we have enough duration
        if total_real_duration >= total_fake_duration:
            print(f"\n   🎯 Reached target duration!")
            break

        audio, sr = read_audio_file(path)
        if audio is None:
            continue

        # Resample
        audio = resample(audio, sr, TARGET_SAMPLE_RATE)
        duration = len(audio) / TARGET_SAMPLE_RATE
        total_real_duration += duration
        files_used += 1

        # Chop into 3-second chunks
        chunks = chop_into_chunks(audio, TARGET_SAMPLE_RATE)
        for j, chunk in enumerate(chunks):
            out_name = f"real_{files_used:04d}_{j:04d}.wav"
            save_wav_chunk(chunk, TARGET_SAMPLE_RATE, os.path.join(OUTPUT_REAL, out_name))
            real_chunk_count += 1

        if files_used % 50 == 0:
            print(f"   Processed {files_used} files... ({total_real_duration:.0f}s / {total_fake_duration:.0f}s)")

    # ==========================================
    # SUMMARY
    # ==========================================
    print(f"\n{'='*60}")
    print(f"DATASET READY!")
    print(f"{'='*60}")
    print(f"   AI chunks : {ai_chunk_count} ({total_fake_duration:.0f}s)")
    print(f"   Real chunks: {real_chunk_count} ({total_real_duration:.0f}s)")
    print(f"   Real files used: {files_used} / {len(real_files)}")
    print(f"   Output: {OUTPUT_AI}/ and {OUTPUT_REAL}/")
    print(f"\n   Next step: python train_audio.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    prepare_dataset()
