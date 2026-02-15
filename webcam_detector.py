import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from models.hybrid_fft import HybridFFT

# ==========================================================
# REAL-TIME WEBCAM DETECTION (Feature 8)
# Live video feed analysis using webcam
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


def load_webcam_model():
    """Load the HybridFFT model for real-time detection."""
    print(f"Loading model on {DEVICE}...")
    model = HybridFFT()
    model.load_state_dict(torch.load("hybrid.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def run_webcam_detection(camera_index=0, process_every_n=5):
    """
    Start real-time webcam deepfake detection.
    
    Args:
        camera_index: Camera device index (0 = default webcam)
        process_every_n: Process every Nth frame (for performance)
    
    Controls:
        - Press 'Q' to quit
        - Press 'S' to save current frame
        - Press '+/-' to adjust processing frequency
    """
    model = load_webcam_model()
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print("\n📹 Webcam Detection Active!")
    print("   Controls:")
    print("   [Q] Quit")
    print("   [S] Save current frame")
    print("   [+/-] Adjust processing speed")
    print("=" * 50)

    frame_count = 0
    current_label = "Analyzing..."
    current_confidence = 0.0
    current_color = (255, 255, 0)  # Yellow while analyzing
    
    # Rolling average for smoother predictions
    recent_scores = []
    max_recent = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()

        # Process every Nth frame
        if frame_count % process_every_n == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(tensor)
                probs = F.softmax(output, dim=1)
                conf, cls = torch.max(probs, dim=1)
                is_ai = cls.item() == 1
                score = conf.item()

            # Track rolling average
            recent_scores.append(1.0 if is_ai else 0.0)
            if len(recent_scores) > max_recent:
                recent_scores.pop(0)

            avg_fake_score = sum(recent_scores) / len(recent_scores)

            if avg_fake_score > 0.6:
                current_label = "DEEPFAKE DETECTED"
                current_confidence = score
                current_color = (0, 0, 255)  # Red (BGR)
            elif avg_fake_score > 0.3:
                current_label = "SUSPICIOUS"
                current_confidence = score
                current_color = (0, 165, 255)  # Orange
            else:
                current_label = "REAL"
                current_confidence = score
                current_color = (0, 255, 0)  # Green

        # === OVERLAY UI ===
        h, w = display_frame.shape[:2]
        
        # Top banner
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

        # Status text
        cv2.putText(display_frame, f"Status: {current_label}",
                     (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, current_color, 2)
        cv2.putText(display_frame, f"Confidence: {current_confidence*100:.1f}%",
                     (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Confidence bar
        bar_width = int((w - 30) * current_confidence)
        cv2.rectangle(display_frame, (15, 70), (15 + bar_width, 75), current_color, -1)
        cv2.rectangle(display_frame, (15, 70), (w - 15, 75), (100, 100, 100), 1)

        # Bottom info
        fps_text = f"Frame: {frame_count} | Every {process_every_n} frames | [Q]uit [S]ave"
        cv2.putText(display_frame, fps_text,
                     (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Border color based on detection
        cv2.rectangle(display_frame, (0, 0), (w-1, h-1), current_color, 3)

        # Show frame
        cv2.imshow("AI Deepfake Detection - Live", display_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            save_path = f"webcam_capture_{frame_count}.jpg"
            cv2.imwrite(save_path, frame)
            print(f"📸 Saved: {save_path}")
        elif key == ord('+') or key == ord('='):
            process_every_n = max(1, process_every_n - 1)
            print(f"⚡ Processing every {process_every_n} frames")
        elif key == ord('-'):
            process_every_n = min(30, process_every_n + 1)
            print(f"🐢 Processing every {process_every_n} frames")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Webcam detection stopped.")


# ==========================================================
# CLI ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    import sys
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_webcam_detection(camera_index=cam_idx)
