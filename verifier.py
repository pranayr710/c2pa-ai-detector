from utils import get_pixel_hash, get_file_hash, get_c2pa_seal
from ai_detector import ai_detect
import os


def verify_image(image_path):
    """Verify image authenticity using cryptographic seal with AI fallback"""

    if not os.path.isfile(image_path):
        return {
            "status": "ERROR",
            "message": f"Image file not found: {image_path}",
            "verified": False
        }

    try:
        seal = get_c2pa_seal(image_path)

        # ------------------------------------------------
        # CASE 1: Seal exists
        # ------------------------------------------------
        if seal:
            if image_path.lower().endswith(".png"):
                current_hash = get_pixel_hash(image_path)
            else:
                current_hash = get_file_hash(image_path)

            # ✅ Seal valid
            if current_hash == seal:
                return {
                    "status": "VERIFIED_REAL",
                    "message": "✅ VERIFIED (Cryptographic proof)",
                    "verified": True,
                    "method": "C2PA_SEAL"
                }

            # ❌ Seal tampered → RUN AI
            result = {
                "status": "TAMPERED",
                "message": "❌ TAMPERED (Seal mismatch)",
                "verified": False,
                "method": "C2PA_SEAL"
            }

            score = ai_detect(image_path)

            if score > 0.7:
                result["ml_verdict"] = "LIKELY_AI"
                result["ml_confidence"] = score
            else:
                result["ml_verdict"] = "LIKELY_REAL"
                result["ml_confidence"] = 1 - score

            result["ml_method"] = "ML_DETECTION"
            return result

        # ------------------------------------------------
        # CASE 2: No seal → AI only
        # ------------------------------------------------
        score = ai_detect(image_path)

        if score > 0.7:
            return {
                "status": "LIKELY_AI",
                "message": f"🤖 AI GENERATED ({score*100:.1f}%)",
                "verified": False,
                "confidence": score,
                "method": "ML_DETECTION"
            }
        else:
            return {
                "status": "LIKELY_REAL",
                "message": f"⚠️ UNVERIFIED REAL ({(1-score)*100:.1f}%)",
                "verified": False,
                "confidence": 1 - score,
                "method": "ML_DETECTION"
            }

    except Exception as e:
        return {
            "status": "ERROR",
            "message": f"Verification error: {str(e)}",
            "verified": False
        }
