import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime

from models.metadata_mlp import MetadataMLP

# ==========================================================
# METADATA FORENSICS (ML-Powered)
# Uses trained MetadataMLP + rule-based findings
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "metadata_model.pth"

# Known AI software signatures
AI_SOFTWARE_SIGNATURES = [
    "adobe firefly", "midjourney", "dall-e", "stable diffusion",
    "dreamstudio", "playground ai", "leonardo ai", "canva ai",
    "comfyui", "automatic1111", "invokeai", "fooocus",
    "runway", "pika", "nightcafe", "wombo", "craiyon"
]

AI_METADATA_KEYS = [
    "prompt", "negative_prompt", "cfg_scale", "sampler",
    "sd_model", "model_hash", "steps", "seed", "ai_generated"
]

_meta_model = None


def load_metadata_model():
    """Load trained MetadataMLP model."""
    global _meta_model
    if _meta_model is None:
        if not os.path.exists(MODEL_PATH):
            print(f"⚠️  Metadata model not found at {MODEL_PATH}")
            print(f"   Run: python train_metadata.py")
            return None

        print(f"Loading Metadata Model (MetadataMLP) on {DEVICE}...")
        model = MetadataMLP(input_size=15)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        _meta_model = model
    return _meta_model


# ==========================================================
# FEATURE EXTRACTION (matches training pipeline exactly)
# ==========================================================
def extract_metadata_features(image_path):
    """Extract 15 numeric features from image metadata."""
    features = np.zeros(15, dtype=np.float32)

    try:
        img = Image.open(image_path)
        stat = os.stat(image_path)
        w, h = img.size

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

        all_values = " ".join(all_tags.values())

        # PNG text chunks
        all_keys_lower = [k.lower() for k in all_tags.keys()]
        if hasattr(img, "text") and img.text:
            for k, v in img.text.items():
                all_keys_lower.append(k.lower())
                all_values += " " + str(v).lower()

        features[0] = 1.0 if "Make" in all_tags else 0.0
        features[1] = 1.0 if "Model" in all_tags else 0.0
        features[2] = 1.0 if "GPSInfo" in all_tags else 0.0
        features[3] = 1.0 if "DateTime" in all_tags else 0.0
        features[4] = 1.0 if "DateTimeOriginal" in all_tags else 0.0
        features[5] = 1.0 if "Software" in all_tags else 0.0
        features[6] = 1.0 if any(sig in all_values for sig in AI_SOFTWARE_SIGNATURES) else 0.0
        features[7] = 1.0 if any(key in " ".join(all_keys_lower) for key in AI_METADATA_KEYS) else 0.0
        features[8] = min(len(all_tags) / 50.0, 1.0)
        features[9] = min(stat.st_size / (10 * 1024 * 1024), 1.0)
        features[10] = 1.0 if w == h else 0.0
        power_of_2 = [256, 512, 768, 1024, 2048, 4096]
        features[11] = 1.0 if (w in power_of_2 and h in power_of_2) else 0.0
        features[12] = min(w / 4096.0, 1.0)
        features[13] = min(h / 4096.0, 1.0)
        features[14] = 1.0 if ("LensModel" in all_tags or "LensMake" in all_tags) else 0.0

    except Exception:
        pass

    return features


# ==========================================================
# METADATA EXTRACTION (for display/findings)
# ==========================================================
def extract_full_metadata(image_path):
    """Extract comprehensive metadata for display."""
    metadata = {
        "file_info": {},
        "exif_data": {},
        "camera_info": {},
        "gps_info": {},
        "software_info": {},
        "raw_tags": {}
    }

    try:
        stat = os.stat(image_path)
        metadata["file_info"] = {
            "filename": os.path.basename(image_path),
            "file_size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "extension": os.path.splitext(image_path)[1].lower()
        }

        img = Image.open(image_path)
        metadata["file_info"]["dimensions"] = f"{img.width}x{img.height}"
        metadata["file_info"]["format"] = img.format

        exif_data = None
        try:
            exif_data = img._getexif()
        except:
            pass

        if exif_data:
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8", errors="ignore")
                    except:
                        value = str(value)

                metadata["raw_tags"][str(tag_name)] = str(value)

                if tag_name in ["Make", "Model", "LensModel", "LensMake"]:
                    metadata["camera_info"][tag_name] = str(value)
                elif tag_name in ["Software", "ProcessingSoftware"]:
                    metadata["software_info"][tag_name] = str(value)
                elif tag_name == "GPSInfo":
                    try:
                        for gps_tag_id, gps_value in value.items():
                            gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                            metadata["gps_info"][str(gps_tag)] = str(gps_value)
                    except:
                        pass
                elif tag_name in ["DateTime", "DateTimeOriginal", "DateTimeDigitized"]:
                    metadata["exif_data"][tag_name] = str(value)

        if hasattr(img, "text") and img.text:
            for key, value in img.text.items():
                metadata["raw_tags"][key] = str(value)
                if any(m in key.lower() for m in ["software", "creator", "tool"]):
                    metadata["software_info"][key] = str(value)

    except Exception as e:
        metadata["error"] = str(e)

    return metadata


# ==========================================================
# RULE-BASED FINDINGS (supplements ML)
# ==========================================================
def generate_findings(metadata):
    """Generate human-readable findings from metadata."""
    findings = []

    if not metadata["camera_info"]:
        findings.append({"severity": "MEDIUM", "finding": "No camera information found",
                         "detail": "Real photos typically contain camera make/model"})

    all_values = " ".join(str(v) for v in metadata["software_info"].values()).lower()
    all_tags = " ".join(str(v) for v in metadata["raw_tags"].values()).lower()

    for sig in AI_SOFTWARE_SIGNATURES:
        if sig in all_values or sig in all_tags:
            findings.append({"severity": "HIGH", "finding": f"AI software detected: {sig}",
                             "detail": "Image likely created with AI tools"})
            break

    if not metadata["exif_data"]:
        findings.append({"severity": "LOW", "finding": "No timestamp information",
                         "detail": "Real camera photos include capture timestamps"})

    if len(metadata["raw_tags"]) < 3:
        findings.append({"severity": "MEDIUM", "finding": "Very sparse metadata",
                         "detail": f"Only {len(metadata['raw_tags'])} tags found"})

    dims = metadata["file_info"].get("dimensions", "")
    if dims:
        w, h = dims.split("x")
        w, h = int(w), int(h)
        if w == h and w in [256, 512, 768, 1024, 2048, 4096]:
            findings.append({"severity": "MEDIUM", "finding": f"Suspicious dimensions: {dims}",
                             "detail": "Power-of-2 square dimensions common in AI images"})

    return findings


# ==========================================================
# MAIN DETECTION
# ==========================================================
def run_metadata_forensics(image_path):
    """
    Main entry point: ML prediction + rule-based findings.
    Combines both for a balanced risk score.
    """
    print("\n🔎 Running Metadata Forensics...")

    # Extract features and metadata
    features = extract_metadata_features(image_path)
    metadata = extract_full_metadata(image_path)
    findings = generate_findings(metadata)

    # === Rule-based risk score ===
    rule_score = 0.0
    for f in findings:
        if f["severity"] == "HIGH":
            rule_score += 0.35  # AI software found = very suspicious
        elif f["severity"] == "MEDIUM":
            rule_score += 0.15  # Missing camera, sparse metadata, etc.
        else:
            rule_score += 0.05
    rule_score = min(rule_score, 1.0)

    # === Camera authenticity bonus ===
    # If image has real camera EXIF data, it's likely real
    has_camera = bool(metadata["camera_info"])
    has_gps = bool(metadata["gps_info"])
    has_timestamps = bool(metadata["exif_data"])
    has_lens = any("Lens" in k for k in metadata.get("camera_info", {}))

    authenticity_indicators = sum([has_camera, has_gps, has_timestamps, has_lens])
    # Each real camera indicator reduces risk
    authenticity_reduction = authenticity_indicators * 0.15

    # === ML prediction ===
    ml_score = 0.5  # neutral default
    model = load_metadata_model()
    if model is not None:
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1)
            ml_score = probs[0][1].item()
        method = "ml_metadata_mlp + rules"
    else:
        method = "rules_only"

    # === Combined score ===
    # 50% rules + 30% ML + 20% authenticity
    risk_score = (
        rule_score * 0.50 +
        ml_score * 0.30 -
        authenticity_reduction * 0.20
    )
    risk_score = max(0.0, min(1.0, risk_score))

    # Label
    if risk_score > 0.6:
        label = "🔴 HIGH RISK — Likely AI Generated"
    elif risk_score > 0.3:
        label = "⚠️ MEDIUM RISK — Suspicious Metadata"
    else:
        label = "🟢 LOW RISK — Appears Authentic"

    # Print results
    print(f"\n   📁 File: {metadata['file_info'].get('filename', 'unknown')}")
    print(f"   📐 Dimensions: {metadata['file_info'].get('dimensions', 'unknown')}")
    print(f"   📷 Camera: {metadata['camera_info'].get('Make', 'N/A')} {metadata['camera_info'].get('Model', '')}")
    print(f"   📍 GPS: {'Yes' if has_gps else 'No'}")
    print(f"   🏷️  EXIF Tags: {len(metadata.get('raw_tags', {}))}")

    print(f"\n   📊 Score Breakdown:")
    print(f"      Rule Score  : {rule_score*100:.1f}%")
    print(f"      ML Score    : {ml_score*100:.1f}%")
    print(f"      Authenticity: {authenticity_indicators}/4 indicators")
    print(f"      Final Risk  : {risk_score*100:.1f}%")
    print(f"\n   Verdict: {label}")

    if findings:
        print(f"\n   📋 Findings ({len(findings)}):")
        for f in findings:
            icon = "🔴" if f["severity"] == "HIGH" else ("⚠️" if f["severity"] == "MEDIUM" else "ℹ️")
            print(f"      {icon} [{f['severity']}] {f['finding']}")

    return {
        "risk_score": risk_score,
        "label": label,
        "findings": findings,
        "metadata": metadata,
        "method": method,
        "score_breakdown": {
            "rule_score": rule_score,
            "ml_score": ml_score,
            "authenticity_indicators": authenticity_indicators
        }
    }


# ==========================================================
# CLI ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = run_metadata_forensics(sys.argv[1])
        print("\n" + "=" * 50)
        print("METADATA FORENSICS RESULT")
        print("=" * 50)
        print(f"Risk   : {result['risk_score']*100:.1f}%")
        print(f"Verdict: {result['label']}")
        print(f"Method : {result['method']}")
        print(f"Issues : {len(result['findings'])}")
        print("=" * 50)
    else:
        print("Usage: python metadata_forensics.py <path_to_image>")
