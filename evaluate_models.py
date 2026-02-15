import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import numpy as np

from dataset_loader import ImageDataset
from models.cnn3 import CNN3
from models.cnn6 import CNN6
from models.efficientnet import EfficientNetB0
from models.hybrid_fft import HybridFFT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

MODELS = {
    "CNN-3": ("cnn3.pth", CNN3),
    "CNN-6": ("cnn6.pth", CNN6),
    "EfficientNet-B0": ("efficientnet.pth", EfficientNetB0),
    "Hybrid": ("hybrid.pth", HybridFFT),
}


# =====================================================
# LOAD DATASET
# =====================================================

dataset = ImageDataset("dataset", split="val")
loader = DataLoader(dataset, batch_size=16, shuffle=False)


# =====================================================
# EVALUATION FUNCTION
# =====================================================

def evaluate_model(model, loader):
    y_true, y_pred = [], []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    return np.array(y_true), np.array(y_pred)


# =====================================================
# METRICS COMPUTATION
# =====================================================

results = {}

for name, (path, model_class) in MODELS.items():
    print(f"\nEvaluating {name}")

    model = model_class().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))

    y_true, y_pred = evaluate_model(model, loader)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)

    results[name] = {
        "accuracy": acc * 100,
        "precision": prec * 100,
        "recall": rec * 100,
        "f1": f1 * 100,
        "fpr": fpr * 100,
        "params": sum(p.numel() for p in model.parameters()) / 1e6
    }


# =====================================================
# PRINT TABLE (PAPER READY)
# =====================================================

print("\n================= MODEL COMPARISON =================")
print(f"{'Model':<18}{'Params(M)':<10}{'Acc%':<8}{'Prec%':<8}{'Recall%':<8}{'F1%':<8}{'FPR%':<8}")

for name, r in results.items():
    print(f"{name:<18}{r['params']:<10.2f}{r['accuracy']:<8.2f}{r['precision']:<8.2f}{r['recall']:<8.2f}{r['f1']:<8.2f}{r['fpr']:<8.2f}")


# =====================================================
# GRAPH 1 — ACCURACY COMPARISON
# =====================================================

models = list(results.keys())
accuracy = [results[m]["accuracy"] for m in models]

plt.figure()
plt.bar(models, accuracy)
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison")
plt.show()


# =====================================================
# GRAPH 2 — ACCURACY vs INFERENCE COST (PARAMS)
# =====================================================

params = [results[m]["params"] for m in models]

plt.figure()
plt.scatter(params, accuracy)
for i, m in enumerate(models):
    plt.text(params[i], accuracy[i], m)
plt.xlabel("Model Parameters (Million)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Model Complexity")
plt.show()


# =====================================================
# GRAPH 3 — FALSE POSITIVE RATE
# =====================================================

fpr = [results[m]["fpr"] for m in models]

plt.figure()
plt.bar(models, fpr)
plt.ylabel("False Positive Rate (%)")
plt.title("False Positive Rate Comparison")
plt.show()
