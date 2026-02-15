# Project Overview: C2PA AI Deepfake Detection & Authentication System

## 1. Executive Summary
This project is a cutting-edge **Hybrid Media Authentication System** designed to combat the proliferation of deepfakes and manipulated media. By combining **C2PA cryptographic provenance** (Coalition for Content Provenance and Authenticity) with state-of-the-art **Ensemble AI Detection**, the system offers a dual-layer defense mechanism. It verifies the source and integrity of media files while simultaneously analyzing them for AI-generated artifacts using multiple deep learning models.

The solution is deployed as a comprehensive toolkit featuring a CLI for batch processing/forensics and a Next.js web application for user-friendly interaction.

## 2. Key Features

### 🛡️ Dual-Layer Verification
- **Layer 1: Cryptographic Sealing (C2PA)**
  - Extracts and verifies digital signatures embedded in image metadata (PNG/JPG).
  -Computes SHA-256 hashes of pixel data to detect pixel-level tampering even if metadata is intact.
  - Returns "VERIFIED REAL" if the cryptographic seal matches the content.

- **Layer 2: AI Ensemble Detection**
  - Activated when no seal is found or verification fails.
  - Utilizes a voting ensemble of 4 distinct Deep Learning models:
    1. **CNN3**: Lightweight convolutional network for rapid screening.
    2. **CNN6**: Deeper architecture for complex feature extraction.
    3. **EfficientNet-B0**: Pre-trained transfer learning model for high accuracy.
    4. **HybridFFT**: Advanced model analyzing frequency domain artifacts (Fourier Transform) to detect GAN upsampling patterns.

### 🎥 Multi-Modal Detection
- **Video Analysis**: Performs temporal frame-by-frame analysis to detect flickering and inconsistencies typical of Deepfake videos.
- **Audio Forensics**: Detects synthetic voice clones and TTS (Text-to-Speech) artifacts.
- **Real-Time Webcam**: Live stream analysis for real-time deepfake detection during video calls.

### 🕵️ Advanced Forensics
- **GAN Fingerprinting**: Identifies the specific generator architecture (e.g., ProGAN, StyleGAN2) used to create a fake image.
- **Metadata Risk Scoring**: Analyzes EXIF/IPTC data for anomalies (e.g., missing device info, software editing traces).

### 📊 Reporting & Batch Processing
- **Automated PDF Reports**: Generates detailed forensic reports with confidence scores, voting breakdowns, and risk assessments.
- **Batch Processing**: Capable of analyzing entire directories of media at once for enterprise-scale auditing.

## 3. Technical Architecture

### Tech Stack
- **Core Logic**: Python 3.9+
- **Deep Learning**: PyTorch, TorchVision
- **Image Processing**: PIL (Pillow), OpenCV
- **Web Frontend**: Next.js (React), TypeScript
- **Metadata/Crypto**: Piexif, Hashlib

### Workflow
1. **Input**: User uploads Image/Video/Audio.
2. **Preprocessing**: Resizing, Normalization, Tensor conversion.
3. **C2PA Check**: Search for `seal` or `hash` in metadata -> Verify integrity.
4. **AI Analysis**:
   - If Seal is valid -> **Pass**.
   - If No Seal -> **Ensemble Model Inference**.
5. **Decision Logic**: Weighted voting system (HybridFFT has 2.0x weight).
6. **Output**: Final Verdict (Real/Fake), Confidence Score, Forensic Report.

## 4. Intended Use Cases
- **Journalism & Media**: Verifying user-generated content before publication.
- **Legal & Forensics**: Authenticating evidence for legal proceedings.
- **Social Media Platforms**: Automated content moderation to flag synthetic media.
- **Identity Verification (KYC)**: Preventing deepfake injection attacks in remote identity verification.
