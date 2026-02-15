# Project Analysis & Specification: SDLC, Agile, Scrum, and Design Thinking

## 1. SDLC (Software Development Life Cycle) Analysis
The development of the **C2PA AI Detection System** follows an **Iterative and Incremental Model**. Given the evolving nature of deepfake technology, a rigid Waterfall model is unsuitable. The project requires continuous refinement of AI models and security protocols.

### Phases Applied to This Project:
1.  **Requirement Analysis**:
    -   **Problem**: Difficulty in distinguishing AI-generated media from authentic content.
    -   **Requirement**: A dual-layered system (AI + Crytography) to verify media.
    -   **Feasibility**: Availability of pre-trained models (EfficientNet) and standards (C2PA).

2.  **System Design**:
    -   **Architecture**: Modular design with separate Model Registry, Verification Logic, and Frontend.
    -   **Data Flow**: Image Input -> C2PA Check -> (If Fail) -> Ensemble AI Voting -> Output.
    -   **Database**: Not strictly relational; uses file-system based model weights and potential local storage for logs.

3.  **Implementation (Coding)**:
    -   **Backend**: Python (PyTorch) for high-performance tensor operations.
    -   **Frontend**: Next.js for a responsive user interface.
    -   **Integration**: Binding Python scripts to the web interface or CLI.

4.  **Testing & Validation**:
    -   **Unit Testing**: Verifying individual functions (e.g., hash computation, metadata extraction).
    -   **Model Validation**: Testing AI models against a validation dataset to ensure accuracy > 90%.
    -   **Integration Testing**: Ensuring the "Ensemble" correctly aggregates votes from all 4 models.

5.  **Deployment**:
    -   Local deployment via CLI (`main.py`) or Web Server (`npm run dev`).

6.  **Maintenance**:
    -   Regular retraining of models (e.g., `train_gan.py`) effectively updates the system against new deepfake generators.

---

## 2. Agile Methodology Specification
The project adheres to **Agile principles**, prioritizing working software over comprehensive documentation and responding to change over following a plan.

### Evidence of Agile in This Project:
-   **Iterative Development**: The presence of multiple model versions (`cnn3.pth`, `cnn6.pth`, `hybrid.pth`) suggests an iterative approach where simpler models were built first, followed by more complex ones.
-   **Modular Components**: Features like `audio_detector.py` and `video_detector.py` are separate modules, allowing them to be developed and integrated independently as "User Stories" in different sprints.
-   **Continuous Improvement**: The "Ensemble" method represents a refactoring and improvement phase where individual weak learners are combined for a strong learner.

---

## 3. Scrum Framework Implementation
If managed under Scrum, the development process would be structured as follows:

### Roles
-   **Product Owner**: Defines the vision (e.g., "We need to detect audio deepfakes now").
-   **Scrum Master**: Removes blockers (e.g., "CUDA errors," "Dataset missing").
-   **Development Team**: Data Scientists (Model training) & Full Stack Devs (Next.js App).

### Artifacts vs. Project Files
-   **Product Backlog**:
    -   *Item 1*: Implement basic CNN detection (Done).
    -   *Item 2*: Add C2PA seal verification (Done).
    -   *Item 3*: Create Web Interface (Done - `app/`).
    -   *Item 4*: Optimize for Real-time Webcam (Done - `webcam_detector.py`).
-   **Sprint Backlog**: The current set of scripts being worked on (e.g., `train_video.py`).

### Events
-   **Sprints**: 2-week cycles.
    -   *Sprint A*: Train CNN3 & CNN6 on image dataset.
    -   *Sprint B*: Implement C2PA seal parsing.
    -   *Sprint C*: Develop HybridFFT model for frequency analysis.

---

## 4. Design Thinking Process
The project formulation demonstrates a human-centric design thinking approach:

### Phase 1: Empathize
-   **Observation**: Users are losing trust in digital media. Journalists cannot verify sources quickly.
-   **Insight**: Users need a "green checkmark" for real content, not just a "fake" warning.

### Phase 2: Define
-   **Point of View**: "A content moderator needs a reliable tool to instantly check if a viral video is real or deepfaked to prevent skepticism."
-   **Problem Statement**: "How might we create a system that combines the certainty of cryptography with the adaptability of AI?"

### Phase 3: Ideate
-   **Brainstorming**:
    -   *Idea 1*: Only use Blockchain? Too slow/expensive.
    -   *Idea 2*: Only use AI? Prone to false positives.
    -   *Idea 3 (Selected)*: **Hybrid Approach**. Use Metadata Seals (C2PA) for known trusted sources, and AI as a fallback for unknown sources.

### Phase 4: Prototype
-   **Low-Fidelity**: Command Line Interface (`main.py`) allowing quick testing of logic without UI overhead.
-   **High-Fidelity**: The Next.js Web App (`app/`) providing a visual dashboard with drag-and-drop functionality.

### Phase 5: Test
-   **Validation**: Running `evaluate_models.py` to get quantitative metrics (Accuracy/Loss).
-   **User Testing**: The explicit addition of `report_generator.py` suggests feedback that users needed "proof" or a paper trail, leading to the PDF report feature.
