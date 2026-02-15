"""
VIDEO AI DETECTION

Uses a DEDICATED video model (VideoCNN) trained on actual video frames.
Falls back to image ensemble if video model isn't trained yet.

Why a dedicated video model?
    Image models (CNN3, EfficientNet, etc.) were trained on clean photos.
    Video frames have compression artifacts (H.264 blocks, motion blur)
    that make real videos look "artificial" to image models.
    
    VideoCNN is trained ON video frames, so compression is normal to it.

Train the video model:
    1. Put real videos in:  dataset/video/real/
    2. Put AI videos in:   dataset/video/ai/
    3. Run: python train_video.py
"""

import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ==========================================================
# CONFIG
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_MODEL_PATH = "video_model.pth"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

_model = None
_model_type = None  # "video" or "ensemble"


# ==========================================================
# MODEL LOADING
# ==========================================================
def load_model():
    """
    Priority: VideoCNN (trained on video) > Image ensemble (fallback)
    """
    global _model, _model_type

    if _model is not None:
        return _model, _model_type

    # Try dedicated video model first
    if os.path.exists(VIDEO_MODEL_PATH):
        from models.video_cnn import VideoCNN
        print(f"✅ Loading dedicated Video Model (VideoCNN) on {DEVICE}...")
        model = VideoCNN()
        model.load_state_dict(torch.load(VIDEO_MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        _model = {"video": model}
        _model_type = "video"
        return _model, _model_type

    # Fallback: load all image models for ensemble
    print("⚠️  Video model not found. Using image model ensemble (less accurate).")
    print("   For better results, run: python train_video.py\n")

    from models.cnn3 import CNN3
    from models.cnn6 import CNN6
    from models.efficientnet import EfficientNetB0
    from models.hybrid_fft import HybridFFT

    configs = [
        ("CNN3",         CNN3,          "cnn3.pth",         1.0),
        ("CNN6",         CNN6,          "cnn6.pth",         1.2),
        ("EfficientNet", EfficientNetB0, "efficientnet.pth", 1.5),
        ("HybridFFT",    HybridFFT,    "hybrid.pth",       2.0),
    ]

    models = {}
    for name, cls, path, weight in configs:
        try:
            m = cls()
            m.load_state_dict(torch.load(path, map_location=DEVICE))
            m.to(DEVICE)
            m.eval()
            models[name] = (m, weight)
            print(f"   ✅ {name}")
        except Exception as e:
            print(f"   ❌ {name}: {e}")

    _model = models
    _model_type = "ensemble"
    return _model, _model_type


# ==========================================================
# FRAME PREDICTION
# ==========================================================
def predict_frame(frame, models, model_type):
    """
    Predict one frame. Returns AI probability (0.0=real, 1.0=fake).
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if model_type == "video":
            # Single dedicated video model
            output = models["video"](tensor)
            probs = F.softmax(output, dim=1)
            return probs[0][1].item()

        else:
            # Ensemble: weighted vote across image models
            ai_weighted = 0.0
            real_weighted = 0.0

            for name, (model, weight) in models.items():
                output = model(tensor)
                probs = F.softmax(output, dim=1)
                ai_prob = probs[0][1].item()

                # Apply bilateral filter for image models to reduce compression noise
                if ai_prob > 0.5:
                    ai_weighted += weight * ai_prob
                else:
                    real_weighted += weight * (1.0 - ai_prob)

            total = ai_weighted + real_weighted + 1e-8
            return ai_weighted / total


# ==========================================================
# FRAME SAMPLING
# ==========================================================
def get_sample_indices(total_frames, target=30):
    """Pick evenly spaced frames across the video."""
    if total_frames <= target:
        return list(range(total_frames))

    step = total_frames / target
    return [int(i * step) for i in range(target)]


# ==========================================================
# MAIN
# ==========================================================
def process_video(video_path, num_frames=30):
    """
    Analyze a video for AI-generated content.
    """
    models, model_type = load_model()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"error": "Could not open video file"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    model_label = "VideoCNN (dedicated)" if model_type == "video" else f"Image Ensemble ({len(models)} models)"

    print(f"\n🎬 Video: {video_path}")
    print(f"   Duration : {duration:.1f}s | FPS: {fps:.0f} | Frames: {total_frames}")
    print(f"   Model    : {model_label}")

    # Get target frames
    target_indices = set(get_sample_indices(total_frames, num_frames))
    print(f"   Sampling : {len(target_indices)} frames\n")

    # Process
    scores = []
    frame_idx = 0
    analyzed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in target_indices:
            ai_score = predict_frame(frame, models, model_type)
            scores.append(ai_score)
            analyzed += 1

            tag = "🔴 AI" if ai_score > 0.5 else "🟢 REAL"
            print(f"   [{analyzed:3d}/{len(target_indices)}] Frame {frame_idx:6d}  {tag}  ({ai_score*100:.1f}%)", end="\r")

        frame_idx += 1

    cap.release()

    if not scores:
        return {"error": "No frames analyzed"}

    print(f"\n\n✅ Analyzed {analyzed} frames")

    # ==========================================================
    # SCORING
    # ==========================================================
    arr = np.array(scores)
    mean_score = float(np.mean(arr))
    median_score = float(np.median(arr))
    fake_count = int(np.sum(arr > 0.5))
    fake_ratio = fake_count / len(arr)

    # Decision: median-based
    is_fake = median_score > 0.5

    # Confidence
    confidence = median_score if is_fake else (1.0 - median_score)

    # Verdict
    if median_score > 0.7:
        verdict = "🔴 AI GENERATED VIDEO"
    elif median_score > 0.5:
        verdict = "⚠️ LIKELY AI GENERATED"
    elif median_score > 0.35:
        verdict = "⚠️ UNCERTAIN"
    else:
        verdict = "🟢 REAL VIDEO"

    # Print
    print(f"\n{'='*55}")
    print(f"  VIDEO ANALYSIS RESULT")
    print(f"{'='*55}")
    print(f"  Verdict        : {verdict}")
    print(f"  Confidence     : {confidence*100:.1f}%")
    print(f"  Mean AI Score  : {mean_score*100:.1f}%")
    print(f"  Median Score   : {median_score*100:.1f}%")
    print(f"  Fake Frames    : {fake_ratio*100:.0f}% ({fake_count}/{len(arr)})")
    print(f"  Frames Sampled : {analyzed}")
    print(f"  Detection Mode : {model_label}")
    print(f"{'='*55}\n")

    return {
        "is_fake": is_fake,
        "confidence": confidence,
        "verdict": verdict,
        "mean_score": mean_score,
        "median_score": median_score,
        "fake_ratio": fake_ratio,
        "frames_analyzed": analyzed,
        "model_type": model_type
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = process_video(sys.argv[1])
        if "error" in result:
            print(f"Error: {result['error']}")
    else:
        print("Usage: python video_detector.py <path_to_video>")
