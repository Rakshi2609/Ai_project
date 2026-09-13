# Trust-AI: Multimodal Consistency & Explainability Layer

**Trust-AI** is a multimodal AI system designed to estimate consistency, anomaly, and confidence across multiple signals (textual, kinematic, acoustic, and visual). 

> **Important:** Trust-AI does NOT attempt to detect lies or determine truthfulness from visual or vocal cues. Instead, it measures cross-modal concordance, behavioral stability, and kinematic deviations against expected reference baselines.

---

## Architecture Overview

```text
Expected Answer + Waypoints       Live Subject Output
         │                               │
         ├───────────────────────────────┤
         ▼                               ▼
 ┌─────────────────┐           ┌─────────────────┐
 │ Linguistic NLI  │           │   Kinematic DTW │
 │ & MiniLM Embeds │           │ Trajectory Eval │
 └───────┬─────────┘           └────────┬────────┘
         │                              │
         ├──────────────┬───────────────┤
         ▼              ▼               ▼
┌─────────────────────────────────────────────────┐
│     Multimodal Trust Fusion & Explainability    │
│  - Consistency Score ∈ [0.0, 1.0]               │
│  - Dynamic Confidence Estimation                │
│  - Modality Anomaly Breakdown                   │
└───────────────────────┬─────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────┐
│      Active Human Feedback & Calibration        │
│  - JSONL Supervisor Verification Logs           │
│  - Temperature / Platt Scaling Calibration      │
└─────────────────────────────────────────────────┘
```

---

## Modalities & Features

1. **Linguistic Alignment (Expected vs. Actual Answer):**
   * Pretrained `sentence-transformers/all-MiniLM-L6-v2` for semantic similarity.
   * Contradiction probability and entailment estimation.
2. **Kinematic Consistency (Expected vs. Actual Movement):**
   * Dynamic Time Warping (DTW) for spatial trajectory drift.
   * Execution latency and motor hesitation detection.
3. **Acoustic Behavior & Prosody:**
   * Fundamental frequency ($F_0$) pitch tracking and standard deviation.
   * Acoustic jitter percentage (vocal tension indicator).
   * Unvoiced pause ratio (hesitation duration).
4. **Visual & Facial Affect:**
   * Action Unit (AU) activation tracking.
   * Facial entropy (emotional volatility).
   * Gaze drift variance.
5. **Explainability & Active Feedback:**
   * Human-interpretable explainability summary.
   * Structured JSONL logging (`trust_ai_feedback_logs.jsonl`) for continuous calibration.

---

## Quickstart

### 1. Installation
```bash
git clone https://github.com/Rakshi2609/Ai_project.git
cd Ai_project
pip install torch transformers sentence-transformers numpy scipy
```

### 2. Run Inference & Verification
```bash
python3 trust_ai_pipeline.py
```

---

## Example Output

```json
{
  "predicted_consistency_score": 0.8581,
  "anomaly_detected": false,
  "system_confidence": 0.8777,
  "explainability_summary": "High multi-modal concordance across text, kinematic, and behavioral channels."
}
```

---

## License
MIT License
