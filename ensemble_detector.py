import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.cnn3 import CNN3
from models.cnn6 import CNN6
from models.efficientnet import EfficientNetB0
from models.hybrid_fft import HybridFFT

# ==========================================================
# MODEL ENSEMBLE VOTING
# Runs all 4 models and takes a weighted majority vote
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

# Model configs: (name, class, weight_file, voting_weight)
# Higher voting_weight = more influence on final decision
MODEL_CONFIGS = [
    ("CNN3",          CNN3,          "cnn3.pth",        1.0),
    ("CNN6",          CNN6,          "cnn6.pth",        1.2),
    ("EfficientNet",  EfficientNetB0, "efficientnet.pth", 1.5),
    ("HybridFFT",     HybridFFT,    "hybrid.pth",      2.0),  # Highest weight (best model)
]

_loaded_models = {}


def load_all_models():
    """Load all 4 models into memory."""
    global _loaded_models
    
    for name, cls, weight_file, _ in MODEL_CONFIGS:
        if name not in _loaded_models:
            try:
                model = cls()
                model.load_state_dict(torch.load(weight_file, map_location=DEVICE))
                model.to(DEVICE)
                model.eval()
                _loaded_models[name] = model
                print(f"   ✅ Loaded {name}")
            except Exception as e:
                print(f"   ❌ Failed to load {name}: {e}")
    
    return _loaded_models


def ensemble_predict(image_path):
    """
    Run all models on the same image and aggregate predictions.
    
    Returns:
        dict: {
            "final_label": str,
            "final_confidence": float,
            "is_fake": bool,
            "model_results": [{ name, label, confidence }],
            "vote_breakdown": { "AI": int, "REAL": int }
        }
    """
    print("\n🗳️  Running Ensemble Voting...")
    
    models = load_all_models()
    
    if not models:
        return {"error": "No models could be loaded"}
    
    # Preprocess image once
    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    
    results = []
    ai_votes = 0.0
    real_votes = 0.0
    total_weight = 0.0
    
    for name, _, _, weight in MODEL_CONFIGS:
        if name not in models:
            continue
        
        model = models[name]
        
        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1)
            confidence, cls = torch.max(probs, dim=1)
            
            is_ai = (cls.item() == 1)
            conf = confidence.item()
        
        label = "🤖 AI GENERATED" if is_ai else "✅ REAL"
        
        if is_ai:
            ai_votes += weight
        else:
            real_votes += weight
        total_weight += weight
        
        results.append({
            "model": name,
            "label": label,
            "confidence": conf,
            "vote_weight": weight
        })
        
        print(f"   {name:15s} → {label} ({conf*100:.1f}%)")
    
    # Final decision based on weighted votes
    is_fake = ai_votes > real_votes
    
    # Weighted confidence
    if is_fake:
        final_confidence = ai_votes / total_weight
    else:
        final_confidence = real_votes / total_weight
    
    final_label = "🤖 AI GENERATED" if is_fake else "✅ VERIFIED REAL"
    
    print(f"\n   📊 Vote: AI={ai_votes:.1f} vs REAL={real_votes:.1f}")
    print(f"   🏆 FINAL: {final_label} ({final_confidence*100:.1f}%)")
    
    return {
        "final_label": final_label,
        "final_confidence": final_confidence,
        "is_fake": is_fake,
        "model_results": results,
        "vote_breakdown": {
            "ai_weighted": ai_votes,
            "real_weighted": real_votes,
            "total_weight": total_weight
        }
    }


# ==========================================================
# CLI ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = ensemble_predict(sys.argv[1])
        print("\n" + "=" * 50)
        print("ENSEMBLE RESULT")
        print("=" * 50)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Prediction: {result['final_label']}")
            print(f"Confidence: {result['final_confidence']*100:.2f}%")
            print(f"Models Used: {len(result['model_results'])}")
        print("=" * 50)
    else:
        print("Usage: python ensemble_detector.py <path_to_image>")
