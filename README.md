# FaceTrust System

FaceTrust System is a real-time biometric authentication prototype that combines face recognition and anti-spoofing for secure access decisions in a live webcam setting.

It is designed for capstone demonstration quality: responsive UI, measurable decision signals, and structured session audit logging.

## Key Capabilities

- Realtime face detection using InsightFace RetinaFace
- Identity matching using ArcFace embeddings and cosine similarity
- Liveness verification using three cues:
  - Blink detection (EAR from MediaPipe Face Mesh)
  - Head motion detection (nose landmark displacement)
  - Pose variation detection (normalized geometry change)
- Session-level CSV audit trail for authentication outcomes
- Interactive registration workflow with multi-sample capture

## System Architecture

```text
Webcam Frame
   |
   v
Face Detection (RetinaFace)
   |
   v
Identity Matching (ArcFace)
   |
   v
Liveness Check (MediaPipe EAR + Motion + Pose)
   |
   v
Decision
   |
   +--> UI Overlay (status, confidence, liveness indicators)
   |
   +--> Session Log (CSV audit trail)
```

Pipeline summary:
1. Capture webcam frame.
2. Detect faces and generate ArcFace embeddings.
3. Match against enrolled embeddings.
4. Evaluate liveness with blink, movement, and pose cues.
5. Produce decision (`AUTHENTICATED`, `SPOOF_DETECTED`, or `UNKNOWN`).
6. Render UI and append audit event when status changes.

## Security Approach

Single-cue liveness systems are often fragile. For example, a printed photo can sometimes trigger minor movement, while a replay video can imitate blink timing.

FaceTrust uses a three-cue strategy:
1. Blink cue: validates real eye dynamics using EAR transitions.
2. Motion cue: tracks natural head/nose movement over time.
3. Pose cue: checks geometric variation less likely from flat artifacts.

A decision is treated as live only when cues are consistent within the recent validity window. This multi-signal gating reduces false acceptance risk compared with single-cue approaches.

## Project Structure

```text
FaceTrust/
├─ main.py
├─ face_recognition.py
├─ anti_spoofing.py
├─ utils.py
├─ session_logger.py
├─ requirements.txt
├─ README.md
├─ dataset/       #Add you data sets here 
├─ embeddings/
│  └─ face_embeddings.npz
├─ logs/
│  └─ session_YYYYMMDD_HHMMSS.csv
└─ models/
```

## Setup (Windows)

Recommended Python version: `3.10` or `3.11`.

1. Create environment:

```powershell
py -3.11 -m venv .venv311
```

2. Activate environment:

```powershell
.\.venv311\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Run application:

```powershell
python main.py
```

## Controls

- `N`: Set active identity name
- `C`: Capture registration burst samples
- `R`: Rebuild embeddings from `dataset/`
- `Q`: Quit application

## Data and Logging

- Captured faces are stored in `dataset/<person_name>/`
- Embeddings are stored in `embeddings/face_embeddings.npz`
- Session audit logs are written to `logs/session_YYYYMMDD_HHMMSS.csv`

Each log event includes:
- ISO 8601 timestamp
- person name
- decision result
- confidence score
- blink cue status
- motion cue status

## Limitations

This project is a strong capstone prototype, not a full commercial PAD (Presentation Attack Detection) product. Current constraints include:

1. 2D RGB camera only, without depth/IR sensing.
2. Performance and robustness depend on lighting quality and camera stability.
3. Domain shift risk (different cameras, extreme angles, occlusions, masks).
4. Local embedding store with no role-based access controls or distributed identity governance.

## Future Work

1. Depth-camera or IR-assisted liveness for stronger anti-replay guarantees.
2. Challenge-response liveness (for example randomized blink patterns).
3. Cloud identity synchronization with secure enrollment and centralized policy control.

## Presentation Notes

For faculty demonstration, emphasize:
1. End-to-end decision transparency (UI + audit logs).
2. Multi-cue liveness rationale and threat model.
3. Practical engineering tradeoffs (latency, CPU-only compatibility, and reliability tuning).
