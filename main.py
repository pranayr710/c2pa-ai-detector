import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import hashlib
import piexif


from models.cnn3 import CNN3
from models.cnn6 import CNN6
from models.efficientnet import EfficientNetB0
from models.hybrid_fft import HybridFFT


# ==========================================================
# DEVICE & TRANSFORMS
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================================
# MODEL REGISTRY
# ==========================================================

MODEL_MAP = {
    "1": ("cnn3", CNN3),
    "2": ("cnn6", CNN6),
    "3": ("efficientnet", EfficientNetB0),
    "4": ("hybrid", HybridFFT),
}


def load_model(choice):
    model_name, model_class = MODEL_MAP[choice]
    model = model_class()
    model.load_state_dict(torch.load(f"{model_name}.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, model_name


# ==========================================================
# ML PREDICTION (LABELS FIXED + CONFIDENCE CLAMPED)
# ==========================================================

def ml_predict(model, image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)

    probs = F.softmax(output, dim=1)
    confidence, cls = torch.max(probs, dim=1)

    confidence = confidence.item()  # RAW confidence (no modification)

    if cls.item() == 1:
        label = "🤖 AI GENERATED"
    else:
        label = "⚠️ UNVERIFIED REAL"

    return label, confidence


# ==========================================================
# SEAL UTILITIES
# ==========================================================
def extract_seal_hash(image_path):
    """
    Extract seal hash from:
    - PNG text metadata
    - JPG EXIF UserComment
    """
    try:
        img = Image.open(image_path)

        # ===== PNG METADATA =====
        if image_path.lower().endswith(".png"):
            if hasattr(img, "text"):
                for key, value in img.text.items():
                    if "seal" in key.lower() or "hash" in key.lower():
                        return value

        # ===== JPG / JPEG EXIF =====
        if image_path.lower().endswith((".jpg", ".jpeg")):
            exif_dict = piexif.load(image_path)
            user_comment = exif_dict["Exif"].get(piexif.ExifIFD.UserComment)

            if user_comment:
                try:
                    return user_comment.decode("utf-8")
                except:
                    return user_comment.decode("latin-1")

        return None

    except Exception as e:
        print(f"[SEAL] Extraction failed: {e}")
        return None



def compute_image_hash(image_path):
    """
    Computes SHA-256 hash of image pixel data.
    """
    img = Image.open(image_path).convert("RGB")
    return hashlib.sha256(img.tobytes()).hexdigest()


# ==========================================================
# MAIN SYSTEM (OPTION 5 ONLY)
# ==========================================================

def main_system(image_path):
    print("\n🔍 Running MAIN SYSTEM verification...")

    # STEP 1 — Check seal
    sealed_hash = extract_seal_hash(image_path)

    if sealed_hash:
        print("🛡 Seal detected")

        current_hash = compute_image_hash(image_path)

        # STEP 2 — Verify seal integrity
        if sealed_hash == current_hash:
            print("\n" + "=" * 50)
            print("FINAL RESULT")
            print("=" * 50)
            print("Method    : CRYPTOGRAPHIC SEAL")
            print("Prediction: ✅ VERIFIED REAL")
            print("=" * 50)
            return
        else:
            print("⚠️ Seal detected but image is TAMPERED")
            print("🔁 Falling back to ML detection...")

    else:
        print("ℹ️ No seal found — using ML detection")

    # STEP 3 — ML fallback (Hybrid model)
    model, name = load_model("4")
    label, confidence = ml_predict(model, image_path)

    print("\n" + "=" * 50)
    print("FINAL RESULT")
    print("=" * 50)
    print(f"Method    : ML DETECTION ({name})")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("=" * 50)


# ==========================================================
# CLI
# ==========================================================

def main():
    print("\n" + "=" * 55)
    print("   C2PA AI DETECTION SYSTEM — SELECT MODE")
    print("=" * 55)
    print("\n  🔬 Single Model Detection:")
    print("    1 - CNN (3 layers)")
    print("    2 - CNN (6 layers)")
    print("    3 - EfficientNet-B0")
    print("    4 - Hybrid CNN + FFT")
    print("\n  🛡️  System Modes:")
    print("    5 - MAIN SYSTEM (Seal + ML)")
    print("    6 - VIDEO DETECTION (+ Temporal Analysis)")
    print("    7 - AUDIO DEEPFAKE DETECTION")
    print("    8 - ENSEMBLE VOTING (All 4 Models)")
    print("    9 - GAN FINGERPRINT ID")
    print("\n  🔎 Forensics:")
    print("   10 - METADATA FORENSICS")
    print("   11 - BATCH PROCESSING (Folder)")
    print("   12 - PDF REPORT GENERATION")
    print("   13 - REAL-TIME WEBCAM DETECTION")
    print("=" * 55)

    choice = input("\nEnter option number: ").strip()

    # Webcam doesn't need a file path
    if choice == "13":
        from webcam_detector import run_webcam_detection
        cam = input("Camera index (default 0): ").strip() or "0"
        run_webcam_detection(camera_index=int(cam))
        return

    # Batch processing needs a folder path
    if choice == "11":
        folder_path = input("\nEnter folder path: ").strip()
        use_ens = input("Use ensemble voting? (y/n): ").strip().lower() == "y"
        from batch_processor import batch_process
        result = batch_process(folder_path, use_ensemble=use_ens)

        # Optionally generate report
        gen_report = input("\nGenerate PDF report? (y/n): ").strip().lower()
        if gen_report == "y":
            from report_generator import generate_report
            generate_report(result)
        return

    # All other options need a file path
    file_path = input("\nEnter file path: ").strip()

    if choice == "5":
        main_system(file_path)
        return

    if choice == "6":
        from video_detector import process_video
        result = process_video(file_path)
        if "error" in result:
            print(f"\n❌ Error: {result['error']}")
        # The new video_detector prints its own detailed summary
        return

    if choice == "7":
        from audio_detector import detect_audio
        result = detect_audio(file_path)
        print("\n" + "=" * 50)
        print("AUDIO DETECTION RESULT")
        print("=" * 50)
        print(f"Score  : {result['audio_score'] * 100:.2f}%")
        print(f"Result : {result['audio_label']}")
        if result['flags']:
            print("Flags  :")
            for f in result['flags']:
                print(f"   ⚡ {f}")
        print("=" * 50)
        return

    if choice == "8":
        from ensemble_detector import ensemble_predict
        result = ensemble_predict(file_path)
        print("\n" + "=" * 50)
        print("ENSEMBLE VOTING RESULT")
        print("=" * 50)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Prediction: {result['final_label']}")
            print(f"Confidence: {result['final_confidence'] * 100:.2f}%")
            print(f"AI Votes  : {result['vote_breakdown']['ai_weighted']:.1f}")
            print(f"Real Votes: {result['vote_breakdown']['real_weighted']:.1f}")
        print("=" * 50)
        return

    if choice == "9":
        from gan_fingerprint import detect_gan_fingerprint
        result = detect_gan_fingerprint(file_path)
        print("\n" + "=" * 50)
        print("GAN FINGERPRINT RESULT")
        print("=" * 50)
        print(f"Generator : {result['generator']}")
        print(f"Confidence: {result['confidence'] * 100:.2f}%")
        if result['all_matches']:
            print("\nTop Matches:")
            for m in result['all_matches'][:3]:
                print(f"   {m['name']:20s} {m['score']*100:.1f}%")
        print("=" * 50)
        return

    if choice == "10":
        from metadata_forensics import run_metadata_forensics
        result = run_metadata_forensics(file_path)
        print("\n" + "=" * 50)
        print("METADATA FORENSICS RESULT")
        print("=" * 50)
        print(f"Risk   : {result['risk_score'] * 100:.1f}%")
        print(f"Verdict: {result['label']}")
        print(f"Issues : {len(result['findings'])}")
        print("=" * 50)
        return

    if choice == "12":
        # Run full analysis then generate report
        print("\n📊 Running full analysis for report generation...")
        from ensemble_detector import ensemble_predict
        from metadata_forensics import run_metadata_forensics
        from report_generator import generate_report

        ens_result = ensemble_predict(file_path)
        meta_result = run_metadata_forensics(file_path)

        # Combine results
        combined = {
            "file": file_path,
            **ens_result,
            "metadata": meta_result.get("metadata", {}),
            "findings": meta_result.get("findings", []),
        }
        generate_report(combined)
        return

    # Standard single-model detection (options 1-4)
    if choice not in MODEL_MAP:
        print("❌ Invalid choice")
        return

    model, model_name = load_model(choice)
    label, confidence = ml_predict(model, file_path)

    print("\n" + "=" * 50)
    print("RESULT")
    print("=" * 50)
    print(f"Model     : {model_name}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()

