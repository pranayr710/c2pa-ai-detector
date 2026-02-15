# 🛡️ C2PA AI Deepfake Detection & Authentication System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Next.js](https://img.shields.io/badge/Next.js-16.0-black.svg)

A cutting-edge **Hybrid Media Authentication System** designed to combat the proliferation of deepfakes and manipulated media. By combining **C2PA cryptographic provenance** (Coalition for Content Provenance and Authenticity) with state-of-the-art **Ensemble AI Detection**, this system offers a dual-layer defense mechanism against synthetic media.

## 📋 Table of Contents

- [Features](#-features)
- [Technical Architecture](#-technical-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
  - [CLI Interface](#cli-interface)
  - [Web Interface](#web-interface)
- [Detection Models](#-detection-models)
- [Use Cases](#-use-cases)
- [Project Structure](#-project-structure)
- [Development](#-development)
- [Documentation](#-documentation)
- [License](#-license)

## ✨ Features

### 🔐 Dual-Layer Verification

#### Layer 1: Cryptographic Sealing (C2PA)
- Extracts and verifies digital signatures embedded in image metadata (PNG/JPG)
- Computes SHA-256 hashes of pixel data to detect pixel-level tampering
- Returns "VERIFIED REAL" if the cryptographic seal matches the content

#### Layer 2: AI Ensemble Detection
- Activated when no seal is found or verification fails
- Utilizes a voting ensemble of **4 distinct Deep Learning models**:
  1. **CNN3**: Lightweight convolutional network for rapid screening
  2. **CNN6**: Deeper architecture for complex feature extraction
  3. **EfficientNet-B0**: Pre-trained transfer learning model for high accuracy
  4. **HybridFFT**: Advanced model analyzing frequency domain artifacts (Fourier Transform) to detect GAN upsampling patterns

### 🎥 Multi-Modal Detection

- **🖼️ Image Analysis**: Ensemble voting with 4 specialized neural networks
- **📹 Video Detection**: Temporal frame-by-frame analysis detecting flickering and inconsistencies
- **🎤 Audio Forensics**: Detects synthetic voice clones and TTS (Text-to-Speech) artifacts
- **📷 Real-Time Webcam**: Live stream analysis for real-time deepfake detection during video calls

### 🕵️ Advanced Forensics

- **GAN Fingerprinting**: Identifies the specific generator architecture (ProGAN, StyleGAN2, etc.)
- **Metadata Risk Scoring**: Analyzes EXIF/IPTC data for anomalies
- **Anomaly Detection**: Missing device info, software editing traces, timestamp inconsistencies

### 📊 Reporting & Automation

- **Automated PDF Reports**: Detailed forensic reports with confidence scores and risk assessments
- **Batch Processing**: Analyze entire directories for enterprise-scale auditing
- **Comprehensive Logging**: Track all analysis results and system decisions

## 🏗️ Technical Architecture

### Tech Stack

**Backend:**
- Python 3.9+
- PyTorch & TorchVision
- PIL (Pillow) & OpenCV
- Piexif for metadata handling
- NumPy for numerical operations

**Frontend:**
- Next.js 16 (React 19)
- TypeScript
- TailwindCSS
- Radix UI Components

### Detection Workflow

```
┌─────────────────┐
│ Media Input     │
│ (Image/Video/   │
│  Audio)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │
│ • Resize        │
│ • Normalize     │
│ • Convert       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ C2PA Check      │
│ • Extract Seal  │
│ • Verify Hash   │
└────────┬────────┘
         │
    ┌────┴────┐
    │ Sealed? │
    └─┬────┬──┘
  Yes │    │ No
      │    │
      ▼    ▼
 ┌────┴────┴─────┐
 │ Hash Match?   │
 ├───────────────┤
 │ YES → ✅ REAL │
 │ NO  → AI Test │
 └───────┬───────┘
         │
         ▼
┌─────────────────┐
│ Ensemble AI     │
│ • CNN3          │
│ • CNN6          │
│ • EfficientNet  │
│ • HybridFFT     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Weighted Voting │
│ (HybridFFT 2x)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Final Verdict   │
│ • Label         │
│ • Confidence    │
│ • Report        │
└─────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- Node.js 18 or higher
- CUDA-capable GPU (optional, for faster inference)

### Backend Setup

1. **Clone the repository**
```bash
git clone https://github.com/pranayr710/c2pa-ai-detector.git
cd c2pa-ai-detector
```

2. **Create a virtual environment**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **Install Python dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pillow opencv-python piexif numpy matplotlib reportlab
```

### Frontend Setup (Optional - for Web UI)

1. **Install Node.js dependencies**
```bash
npm install
```

2. **Run the development server**
```bash
npm run dev
```

The web interface will be available at `http://localhost:3000`

## 💻 Usage

### CLI Interface

Run the main detection system:

```bash
python main.py
```

You'll see an interactive menu with the following options:

#### 🔬 Single Model Detection
1. **CNN (3 layers)** - Lightweight and fast
2. **CNN (6 layers)** - Balanced accuracy/speed
3. **EfficientNet-B0** - High accuracy transfer learning
4. **Hybrid CNN + FFT** - Frequency domain analysis

#### 🛡️ System Modes
5. **MAIN SYSTEM** - Cryptographic seal verification + ML fallback
6. **VIDEO DETECTION** - Temporal frame analysis
7. **AUDIO DEEPFAKE DETECTION** - Voice clone detection
8. **ENSEMBLE VOTING** - All 4 models combined

#### 🔎 Forensics & Analysis
9. **GAN FINGERPRINT ID** - Identify specific GAN architecture
10. **METADATA FORENSICS** - EXIF/IPTC anomaly detection
11. **BATCH PROCESSING** - Process entire folders
12. **PDF REPORT GENERATION** - Generate forensic reports
13. **REAL-TIME WEBCAM DETECTION** - Live deepfake detection

### Web Interface

1. Start the Next.js development server:
```bash
npm run dev
```

2. Open your browser to `http://localhost:3000`

3. Upload media files through the drag-and-drop interface

4. View real-time analysis results with visual confidence metrics

### Example CLI Usage

**Verify a single image:**
```bash
python main.py
# Select option 5 (MAIN SYSTEM)
# Enter image path: path/to/image.jpg
```

**Batch process a folder:**
```bash
python main.py
# Select option 11 (BATCH PROCESSING)
# Enter folder path: path/to/folder
# Use ensemble voting? (y/n): y
# Generate PDF report? (y/n): y
```

**Real-time webcam detection:**
```bash
python main.py
# Select option 13 (REAL-TIME WEBCAM)
# Camera index (default 0): 0
```

## 🧠 Detection Models

### Model Performance

| Model | Parameters | Accuracy | Speed | Use Case |
|-------|-----------|----------|-------|----------|
| CNN3 | ~100K | 85% | Very Fast | Quick screening |
| CNN6 | ~500K | 90% | Fast | Balanced detection |
| EfficientNet-B0 | ~5M | 95% | Medium | High accuracy |
| HybridFFT | ~2.8M | 93% | Medium | GAN artifacts |

### Training Scripts

All models can be retrained with your own datasets:

```bash
# Train image detection models
python train_all.py

# Train specific models
python train_gan.py        # GAN fingerprinting
python train_audio.py      # Audio deepfake detection
python train_video.py      # Video detection
python train_metadata.py   # Metadata analysis
```

**Note:** Training datasets are **not included** in this repository due to size constraints. You'll need to provide your own datasets in the following structure:

```
dataset/
├── ai/      # AI-generated images
└── real/    # Real images
```

## 🎯 Use Cases

- **📰 Journalism & Media**: Verify user-generated content before publication
- **⚖️ Legal & Forensics**: Authenticate evidence for legal proceedings
- **🔒 Social Media Platforms**: Automated content moderation
- **🆔 Identity Verification (KYC)**: Prevent deepfake injection attacks
- **🎓 Education**: Teach students about AI-generated content
- **🏢 Enterprise Security**: Verify internal communications

## 📁 Project Structure

```
c2pa-ai-detector/
├── app/                    # Next.js web application
├── components/             # React UI components
├── models/                 # PyTorch model architectures
│   ├── cnn3.py
│   ├── cnn6.py
│   ├── efficientnet.py
│   └── hybrid_fft.py
├── documentation/          # Project documentation
│   ├── project_overview.md
│   └── sdlc_analysis.md
├── scripts/               # Utility scripts
├── main.py                # CLI entry point
├── ensemble_detector.py   # Ensemble voting logic
├── video_detector.py      # Video analysis
├── audio_detector.py      # Audio analysis
├── metadata_forensics.py  # Metadata analysis
├── gan_fingerprint.py     # GAN identification
├── seal_verifier.py       # C2PA seal verification
├── batch_processor.py     # Batch processing
├── report_generator.py    # PDF report generation
├── webcam_detector.py     # Real-time detection
├── train_*.py             # Training scripts
├── *.pth                  # Pre-trained model weights
└── README.md              # This file
```

## 🛠️ Development

### SDLC Approach

This project follows an **Iterative and Incremental Model** with **Agile Methodology** and **Scrum Framework**. For detailed information about the development process, see:

- [Project Overview](documentation/project_overview.md)
- [SDLC Analysis](documentation/sdlc_analysis.md)

### Design Thinking

The system was designed using a **human-centric approach**:
1. **Empathize**: Users need trust in digital media
2. **Define**: Combine cryptography certainty with AI adaptability
3. **Ideate**: Hybrid seal + ML approach
4. **Prototype**: CLI for testing, Web UI for production
5. **Test**: Continuous validation with real-world datasets

### Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Documentation

- **[Project Overview](documentation/project_overview.md)** - Comprehensive system architecture and features
- **[SDLC Analysis](documentation/sdlc_analysis.md)** - Development methodology and design thinking process

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **C2PA Initiative** for the content provenance standard
- **PyTorch Team** for the deep learning framework
- **EfficientNet** authors for the pre-trained models
- **Next.js Team** for the excellent web framework

## 📞 Contact

**Developer**: Pranay R  
**GitHub**: [@pranayr710](https://github.com/pranayr710)

---

<div align="center">

**⚠️ Disclaimer**: This tool is for research and verification purposes only. Always use multiple verification methods for critical content authentication.

Made with ❤️ for a safer digital world

</div>
