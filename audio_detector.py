import os
import sys
import wave
import subprocess
import tempfile
import numpy as np
import torch
import torch.nn.functional as F

from models.audio_cnn import AudioCNN

# ==========================================================
# AUDIO DEEPFAKE DETECTION (ML-Powered)
# Uses trained AudioCNN on Mel-spectrograms
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "audio_model.pth"

# Spectrogram settings (must match training)
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
SPEC_SIZE = 128

_audio_model = None


def load_audio_model():
    """Load the trained AudioCNN model."""
    global _audio_model
    if _audio_model is None:
        if not os.path.exists(MODEL_PATH):
            print(f"⚠️  Audio model not found at {MODEL_PATH}")
            print(f"   Run: python train_audio.py")
            return None
        print(f"Loading Audio Model (AudioCNN) on {DEVICE}...")
        model = AudioCNN()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        _audio_model = model
    return _audio_model


# ==========================================================
# AUDIO EXTRACTION
# ==========================================================
def extract_audio_from_video(video_path, output_path=None):
    """Extract audio track from video using ffmpeg."""
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".wav")
    try:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE), "-ac", "1",
            "-y", output_path
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)
        return output_path
    except FileNotFoundError:
        print("⚠️  ffmpeg not found. Install ffmpeg for audio extraction.")
        return None
    except Exception as e:
        print(f"⚠️  Audio extraction failed: {e}")
        return None


# ==========================================================
# SPECTROGRAM GENERATION (matches training pipeline)
# ==========================================================
def read_wav(path):
    """Read WAV file as float32 numpy array."""
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

            if sr != SAMPLE_RATE and sr > 0:
                ratio = SAMPLE_RATE / sr
                new_len = int(len(audio) * ratio)
                audio = np.interp(np.linspace(0, len(audio)-1, new_len), np.arange(len(audio)), audio)

            return audio
    except Exception as e:
        print(f"   ⚠️ Could not read audio: {e}")
        return None


def _create_mel_filterbank(n_fft_bins, n_mels, sample_rate):
    """Create triangular Mel filterbank."""
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(sample_rate / 2), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft_bins - 1) * 2 * hz_points / sample_rate).astype(int)

    filters = np.zeros((n_mels, n_fft_bins))
    for i in range(n_mels):
        left, center, right = bin_points[i], bin_points[i+1], bin_points[i+2]
        for j in range(left, center):
            if center > left:
                filters[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                filters[i, j] = (right - j) / (right - center)
    return filters


def create_mel_spectrogram(audio):
    """Convert audio to Mel-spectrogram tensor."""
    if audio is None or len(audio) < N_FFT:
        return None

    # Fixed 3-second window
    target_len = SAMPLE_RATE * 3
    if len(audio) > target_len:
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
        spectrogram[:, i] = np.abs(np.fft.rfft(frame))

    # Mel filterbank
    mel_filters = _create_mel_filterbank(N_FFT // 2 + 1, N_MELS, SAMPLE_RATE)
    mel_spec = mel_filters @ spectrogram
    mel_spec = np.log1p(mel_spec)

    # Resize
    from PIL import Image
    mel_img = Image.fromarray(mel_spec.astype(np.float32))
    mel_img = mel_img.resize((SPEC_SIZE, SPEC_SIZE))
    mel_spec = np.array(mel_img)

    # Normalize
    if mel_spec.max() > mel_spec.min():
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())

    return mel_spec


# ==========================================================
# DETECTION (ML-based)
# ==========================================================
def detect_audio(video_path):
    """
    Main entry point for audio deepfake detection.
    Uses trained AudioCNN model on Mel-spectrograms.
    """
    print("\n🎵 Running Audio Deepfake Detection (ML)...")

    model = load_audio_model()

    # Extract audio
    wav_path = extract_audio_from_video(video_path)
    if wav_path is None:
        return {
            "audio_score": 0.5,
            "audio_label": "⚠️ NO AUDIO TRACK FOUND",
            "flags": ["Could not extract audio — ffmpeg may not be installed"],
            "method": "none"
        }

    try:
        audio_data = read_wav(wav_path)
        if audio_data is None:
            return {
                "audio_score": 0.5,
                "audio_label": "⚠️ COULD NOT READ AUDIO",
                "flags": [],
                "method": "none"
            }

        # If model is not trained yet, fall back to heuristic
        if model is None:
            print("   ⚠️ Model not trained. Using heuristic analysis.")
            return _heuristic_detect(audio_data)

        # === ML DETECTION ===
        # Process audio in 3-second chunks and average predictions
        chunk_size = SAMPLE_RATE * 3
        chunks = []
        for i in range(0, len(audio_data) - chunk_size + 1, chunk_size):
            chunks.append(audio_data[i:i + chunk_size])

        if not chunks:
            chunks = [audio_data]  # Use whatever we have

        ai_probs = []
        for i, chunk in enumerate(chunks[:20]):  # Max 20 chunks (60 seconds)
            spec = create_mel_spectrogram(chunk)
            if spec is None:
                continue

            tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(tensor)
                probs = F.softmax(output, dim=1)
                ai_prob = probs[0][1].item()
                ai_probs.append(ai_prob)

            print(f"   Chunk {i+1}: {'🔴 AI' if ai_prob > 0.5 else '🟢 REAL'} ({ai_prob:.2f})", end="\r")

        if not ai_probs:
            return {
                "audio_score": 0.5,
                "audio_label": "⚠️ COULD NOT ANALYZE AUDIO",
                "flags": [],
                "method": "ml"
            }

        # Aggregate
        mean_ai_prob = np.mean(ai_probs)
        is_ai = mean_ai_prob > 0.5

        if mean_ai_prob > 0.7:
            label = "🔴 AI GENERATED AUDIO"
        elif mean_ai_prob > 0.5:
            label = "⚠️ SUSPICIOUS AUDIO"
        else:
            label = "🟢 LIKELY REAL AUDIO"

        flags = []
        if mean_ai_prob > 0.7:
            flags.append("High AI probability across audio chunks")
        if np.std(ai_probs) < 0.1 and mean_ai_prob > 0.5:
            flags.append("Consistently AI-like across all segments")

        print(f"\n   Result: {label} ({mean_ai_prob*100:.1f}%)")

        return {
            "audio_score": mean_ai_prob,
            "audio_label": label,
            "flags": flags,
            "chunks_analyzed": len(ai_probs),
            "method": "ml_audio_cnn"
        }

    finally:
        try:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except:
            pass


def _heuristic_detect(audio_data):
    """Fallback heuristic detection when model is not trained."""
    audio = audio_data.astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    fft_data = np.fft.rfft(audio)
    magnitudes = np.abs(fft_data)

    # Spectral flatness
    geometric_mean = np.exp(np.mean(np.log(magnitudes + 1e-10)))
    arithmetic_mean = np.mean(magnitudes) + 1e-10
    flatness = geometric_mean / arithmetic_mean

    # Zero crossing rate
    zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))

    score = 0.0
    flags = []
    if flatness < 0.1:
        score += 0.3
        flags.append("Low spectral flatness (too clean)")
    if zcr < 0.05:
        score += 0.2
        flags.append("Low zero-crossing rate")

    score = min(score, 1.0)
    label = "🔴 AI GENERATED AUDIO" if score > 0.5 else ("⚠️ SUSPICIOUS" if score > 0.3 else "🟢 LIKELY REAL")

    return {
        "audio_score": score,
        "audio_label": label,
        "flags": flags,
        "method": "heuristic_fallback"
    }


# ==========================================================
# CLI ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = detect_audio(sys.argv[1])
        print("\n" + "=" * 50)
        print("AUDIO ANALYSIS RESULT")
        print("=" * 50)
        print(f"Score  : {result['audio_score']*100:.2f}%")
        print(f"Result : {result['audio_label']}")
        print(f"Method : {result['method']}")
        if result.get('flags'):
            for f in result['flags']:
                print(f"   ⚡ {f}")
        print("=" * 50)
    else:
        print("Usage: python audio_detector.py <path_to_video_or_audio>")
