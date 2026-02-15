"""
ML Detector Wrapper (FINAL FIXED VERSION)

Label mapping:
0 = REAL
1 = AI
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from typing import Tuple, Optional

from models.efficientnet import EfficientNetB0


class MLDetectorModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load YOUR trained EfficientNet model
        self.model = EfficientNetB0()
        self.model.load_state_dict(
            torch.load("efficientnet.pth", map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        # SAME transform as training (CRITICAL)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def detect(self, image_path: str) -> Tuple[str, float]:
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(image_tensor)

            probs = F.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)

            predicted_class = int(predicted_class.item())
            confidence = float(confidence.item())

            # FINAL, CORRECT DECISION
            if predicted_class == 1:
                return "LIKELY_AI", confidence
            else:
                return "LIKELY_REAL", confidence

        except Exception as e:
            print(f"[ML ERROR] {e}")
            return "UNKNOWN", 0.0


# Singleton pattern
_ml_detector: Optional[MLDetectorModel] = None


def get_ml_detector() -> MLDetectorModel:
    global _ml_detector
    if _ml_detector is None:
        _ml_detector = MLDetectorModel()
    return _ml_detector


def detect_ai_likelihood(image_path: str) -> Tuple[str, float]:
    detector = get_ml_detector()
    return detector.detect(image_path)
