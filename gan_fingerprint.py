import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.gan_classifier import GANClassifier

# ==========================================================
# GAN FINGERPRINT IDENTIFICATION (ML-Powered)
# Uses trained GANClassifier to identify AI generators
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "gan_model.pth"
CLASS_MAP_PATH = "gan_classes.txt"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

_gan_model = None
_gan_classes = None


def load_gan_model():
    """Load the trained GAN classifier and class map."""
    global _gan_model, _gan_classes

    if _gan_model is None:
        # Load class names
        if os.path.exists(CLASS_MAP_PATH):
            with open(CLASS_MAP_PATH, "r") as f:
                _gan_classes = [line.strip() for line in f if line.strip()]
        else:
            print(f"⚠️  Class map not found at {CLASS_MAP_PATH}")
            return None, None

        if not os.path.exists(MODEL_PATH):
            print(f"⚠️  GAN model not found at {MODEL_PATH}")
            print(f"   Run: python train_gan.py")
            return None, None

        print(f"Loading GAN Classifier on {DEVICE}...")
        num_classes = len(_gan_classes)
        model = GANClassifier(num_classes=num_classes)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        _gan_model = model

    return _gan_model, _gan_classes


def detect_gan_fingerprint(image_path):
    """
    Main entry point for GAN fingerprint identification.
    Uses trained model or falls back to FFT heuristics.
    """
    print("\n🔬 Running GAN Fingerprint Analysis...")

    model, classes = load_gan_model()

    if model is None or classes is None:
        print("   ⚠️ Model not trained. Using FFT heuristic analysis.")
        return _heuristic_detect(image_path)

    # === ML DETECTION ===
    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)

    # Get all class probabilities
    all_matches = []
    for idx, cls_name in enumerate(classes):
        prob = probs[0][idx].item()
        all_matches.append({
            "name": cls_name,
            "score": prob,
            "description": f"Detected as {cls_name}"
        })

    # Sort by probability
    all_matches.sort(key=lambda x: x["score"], reverse=True)

    best_match = all_matches[0]
    generator = best_match["name"]
    confidence = best_match["score"]

    # Print results
    print(f"\n   🏆 Identified Generator: {generator} ({confidence*100:.1f}%)")
    print(f"\n   📋 All Predictions:")
    for m in all_matches:
        bar = "█" * int(m["score"] * 20) + "░" * (20 - int(m["score"] * 20))
        print(f"      {m['name']:20s} {bar} {m['score']*100:.1f}%")

    return {
        "generator": generator,
        "confidence": confidence,
        "all_matches": all_matches,
        "method": "ml_gan_classifier"
    }


def _heuristic_detect(image_path):
    """Fallback: FFT-based heuristic detection."""
    img = Image.open(image_path).convert("L")
    img = img.resize((256, 256))
    pixels = np.array(img, dtype=np.float32)

    fft = np.fft.fft2(pixels)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)

    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2
    radius = min(h, w) // 4
    y, x = np.ogrid[:h, :w]
    center_mask = ((x - center_w)**2 + (y - center_h)**2) <= radius**2

    low_energy = np.sum(magnitude[center_mask] ** 2)
    total_energy = np.sum(magnitude ** 2) + 1e-10
    hf_ratio = (total_energy - low_energy) / total_energy

    # Simple heuristic matching
    generators = [
        ("Midjourney",         (0.15, 0.35)),
        ("DALL-E",             (0.05, 0.20)),
        ("Stable Diffusion",   (0.10, 0.30)),
        ("StyleGAN",           (0.20, 0.45)),
        ("Real Photo",         (0.30, 0.60)),
    ]

    all_matches = []
    for name, (lo, hi) in generators:
        if lo <= hf_ratio <= hi:
            score = 0.7
        else:
            dist = min(abs(hf_ratio - lo), abs(hf_ratio - hi))
            score = max(0, 0.7 - dist * 3)
        all_matches.append({"name": name, "score": score, "description": f"FFT analysis ({name})"})

    all_matches.sort(key=lambda x: x["score"], reverse=True)

    return {
        "generator": all_matches[0]["name"],
        "confidence": all_matches[0]["score"],
        "all_matches": all_matches,
        "method": "heuristic_fallback"
    }


# ==========================================================
# CLI ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = detect_gan_fingerprint(sys.argv[1])
        print("\n" + "=" * 50)
        print("GAN FINGERPRINT RESULT")
        print("=" * 50)
        print(f"Generator : {result['generator']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"Method    : {result['method']}")
        print("=" * 50)
    else:
        print("Usage: python gan_fingerprint.py <path_to_image>")
