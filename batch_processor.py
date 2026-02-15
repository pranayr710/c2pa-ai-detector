import os
import time
from datetime import datetime

# ==========================================================
# BATCH PROCESSING (Feature 6)
# Process entire folders of images/videos
# ==========================================================

SUPPORTED_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
SUPPORTED_VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}


def scan_folder(folder_path):
    """
    Scan a folder and categorize files into images and videos.
    """
    images = []
    videos = []
    skipped = []

    for root, dirs, files in os.walk(folder_path):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()

            if ext in SUPPORTED_IMAGE_EXTS:
                images.append(fpath)
            elif ext in SUPPORTED_VIDEO_EXTS:
                videos.append(fpath)
            else:
                skipped.append(fpath)

    return images, videos, skipped


def batch_process(folder_path, use_ensemble=False):
    """
    Process all images and videos in a folder.
    
    Args:
        folder_path: Path to folder containing files
        use_ensemble: If True, use ensemble voting; else use HybridFFT only
        
    Returns:
        dict with summary and per-file results
    """
    print(f"\n📂 Batch Processing: {folder_path}")
    print("=" * 60)

    images, videos, skipped = scan_folder(folder_path)
    
    print(f"   Found: {len(images)} images, {len(videos)} videos, {len(skipped)} skipped")
    
    results = []
    start_time = time.time()

    # === Process Images ===
    if images:
        print(f"\n🖼️  Processing {len(images)} images...")
        
        if use_ensemble:
            from ensemble_detector import ensemble_predict
            for i, img_path in enumerate(images, 1):
                print(f"\n   [{i}/{len(images)}] {os.path.basename(img_path)}")
                try:
                    result = ensemble_predict(img_path)
                    results.append({
                        "file": img_path,
                        "type": "IMAGE",
                        "is_fake": result.get("is_fake", False),
                        "confidence": result.get("final_confidence", 0.0),
                        "method": "ensemble",
                        "label": result.get("final_label", "UNKNOWN"),
                        "error": None
                    })
                except Exception as e:
                    results.append({
                        "file": img_path,
                        "type": "IMAGE",
                        "is_fake": None,
                        "confidence": 0.0,
                        "method": "ensemble",
                        "label": "ERROR",
                        "error": str(e)
                    })
        else:
            # Use main system (seal + ML)
            import torch
            import torch.nn.functional as F
            from PIL import Image
            from torchvision import transforms
            from models.hybrid_fft import HybridFFT

            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            TRANSFORM = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            model = HybridFFT()
            model.load_state_dict(torch.load("hybrid.pth", map_location=DEVICE))
            model.to(DEVICE)
            model.eval()

            for i, img_path in enumerate(images, 1):
                print(f"   [{i}/{len(images)}] {os.path.basename(img_path)}", end="")
                try:
                    image = Image.open(img_path).convert("RGB")
                    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        output = model(tensor)
                        probs = F.softmax(output, dim=1)
                        conf, cls = torch.max(probs, dim=1)
                        is_ai = cls.item() == 1
                        score = conf.item()

                    label = "🤖 AI" if is_ai else "✅ REAL"
                    print(f" → {label} ({score*100:.1f}%)")

                    results.append({
                        "file": img_path,
                        "type": "IMAGE",
                        "is_fake": is_ai,
                        "confidence": score,
                        "method": "hybrid_fft",
                        "label": label,
                        "error": None
                    })
                except Exception as e:
                    print(f" → ❌ ERROR")
                    results.append({
                        "file": img_path,
                        "type": "IMAGE",
                        "is_fake": None,
                        "confidence": 0.0,
                        "method": "hybrid_fft",
                        "label": "ERROR",
                        "error": str(e)
                    })

    # === Process Videos ===
    if videos:
        print(f"\n🎬 Processing {len(videos)} videos...")
        from video_detector import process_video

        for i, vid_path in enumerate(videos, 1):
            print(f"\n   [{i}/{len(videos)}] {os.path.basename(vid_path)}")
            try:
                result = process_video(vid_path)
                if "error" in result:
                    raise Exception(result["error"])

                label = "🔴 FAKE" if result["is_fake"] else "🟢 REAL"
                results.append({
                    "file": vid_path,
                    "type": "VIDEO",
                    "is_fake": result["is_fake"],
                    "confidence": result["confidence"],
                    "method": "hybrid_fft_video",
                    "label": label,
                    "error": None
                })
            except Exception as e:
                results.append({
                    "file": vid_path,
                    "type": "VIDEO",
                    "is_fake": None,
                    "confidence": 0.0,
                    "method": "hybrid_fft_video",
                    "label": "ERROR",
                    "error": str(e)
                })

    elapsed = time.time() - start_time

    # === Summary ===
    total = len(results)
    fake_count = sum(1 for r in results if r["is_fake"] is True)
    real_count = sum(1 for r in results if r["is_fake"] is False)
    error_count = sum(1 for r in results if r["is_fake"] is None)

    summary = {
        "folder": folder_path,
        "total_files": total,
        "fake_count": fake_count,
        "real_count": real_count,
        "error_count": error_count,
        "elapsed_seconds": round(elapsed, 2),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }

    # Print summary table
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"   Total Files  : {total}")
    print(f"   🔴 Fake      : {fake_count}")
    print(f"   🟢 Real      : {real_count}")
    print(f"   ❌ Errors     : {error_count}")
    print(f"   ⏱️  Time      : {elapsed:.1f}s")
    print("=" * 60)

    # Detailed table
    print(f"\n{'File':<40} {'Type':<8} {'Result':<12} {'Confidence':<12}")
    print("-" * 72)
    for r in results:
        fname = os.path.basename(r["file"])[:38]
        print(f"{fname:<40} {r['type']:<8} {r['label']:<12} {r['confidence']*100:.1f}%")

    return summary


# ==========================================================
# CLI ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        use_ens = "--ensemble" in sys.argv
        batch_process(folder, use_ensemble=use_ens)
    else:
        print("Usage: python batch_processor.py <folder_path> [--ensemble]")
