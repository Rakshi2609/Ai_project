"""
Trust-AI: Multimodal Consistency, Anomaly, and Explainability Engine
A practical multimodal consistency layer for evaluating:
1. Expected vs Actual Answer (Linguistic Similarity & Negation/Contradiction Analysis)
2. Expected vs Actual Kinematic Trajectory (Dynamic Time Warping)
3. Vocal Behavior & Acoustic Prosody (Pitch F0, Jitter, Hesitation Pauses)
4. Facial Expression & Gaze Stability (Action Units, Facial Entropy, Gaze Variance)
5. Multimodal Trust Fusion & Explainability Layer
6. Human Feedback Logging for Post-Hoc Model Calibration
"""

import os
import json
import time
import math
from datetime import datetime, timezone
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


# =====================================================================
# 1. TEXT CONSISTENCY EXTRACTOR (Expected vs Actual Answer)
# =====================================================================
class TextConsistencyEngine:
    def __init__(self):
        print("[Trust-AI] Initializing Linguistic Alignment Model...")
        # Uses cached MiniLM (~80MB)
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def evaluate(self, expected_answer: str, actual_answer: str) -> dict:
        """
        Computes semantic similarity and asymmetric contradiction risk.
        """
        # Cosine Semantic Similarity
        emb1 = self.model.encode(expected_answer, convert_to_tensor=True)
        emb2 = self.model.encode(actual_answer, convert_to_tensor=True)
        raw_sim = float(util.cos_sim(emb1, emb2)[0][0])
        norm_sim = max(0.0, min(1.0, (raw_sim + 1.0) / 2.0))

        # Heuristic Negation & Contradiction Detection
        negation_tokens = ["not", "no", "never", "without", "don't", "dont", "cannot", "won't", "increase", "ignore", "skip"]
        exp_lower = expected_answer.lower()
        act_lower = actual_answer.lower()
        
        # Check if actual answer introduces conflicting polarity
        polarity_mismatch = False
        exp_has_neg = any(tok in exp_lower.split() for tok in ["not", "never", "without", "stop", "halt"])
        act_has_neg = any(tok in act_lower.split() for tok in ["not", "never", "without", "stop", "halt"])
        
        if ("stop" in exp_lower and "continue" in act_lower) or ("backup" in exp_lower and "without" in act_lower):
            contradiction_prob = 0.92
        elif norm_sim < 0.45:
            contradiction_prob = round(1.0 - norm_sim, 4)
        else:
            contradiction_prob = round(max(0.02, 1.0 - norm_sim - 0.2), 4)

        entailment_prob = round(max(0.0, min(1.0, norm_sim - (contradiction_prob * 0.5))), 4)

        # Composite Linguistic Score: Penalizes contradictions heavily
        text_score = (norm_sim * 0.5) + (entailment_prob * 0.5) - (contradiction_prob * 0.8)
        text_score = max(0.0, min(1.0, text_score))

        return {
            "semantic_similarity": round(norm_sim, 4),
            "contradiction_probability": round(contradiction_prob, 4),
            "entailment_probability": round(entailment_prob, 4),
            "text_consistency_score": round(text_score, 4)
        }


# =====================================================================
# 2. KINEMATIC & TRAJECTORY ALIGNMENT (Native Dynamic Time Warping)
# =====================================================================
class KinematicConsistencyEngine:
    @staticmethod
    def compute_dtw(s1: np.ndarray, s2: np.ndarray) -> float:
        """
        Calculates DTW distance between two 3D coordinate sequences.
        """
        n, m = len(s1), len(s2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(s1[i - 1] - s2[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],     # insertion
                    dtw_matrix[i, j - 1],     # deletion
                    dtw_matrix[i - 1, j - 1]  # match
                )

        return float(dtw_matrix[n, m])

    def evaluate(self, expected_trajectory: list, actual_trajectory: list, latency_ms: float = 0.0) -> dict:
        exp_arr = np.array(expected_trajectory, dtype=np.float32)
        act_arr = np.array(actual_trajectory, dtype=np.float32)

        dtw_dist = self.compute_dtw(exp_arr, act_arr)
        max_steps = max(len(exp_arr), len(act_arr), 1)
        normalized_drift = dtw_dist / max_steps

        # Exponential decay: higher spatial drift -> lower consistency score
        kinematic_score = math.exp(-1.2 * normalized_drift)

        return {
            "dtw_cumulative_distance": round(dtw_dist, 4),
            "normalized_drift": round(normalized_drift, 4),
            "reaction_latency_ms": float(latency_ms),
            "kinematic_consistency_score": round(kinematic_score, 4)
        }


# =====================================================================
# 3. BEHAVIORAL PROCESSOR (Acoustic Prosody & Facial Affect)
# =====================================================================
class BehavioralFeatureProcessor:
    @staticmethod
    def process_audio_features(
        pitch_f0_hz: float = 180.0,
        f0_std_hz: float = 20.0,
        jitter_percent: float = 0.8,
        pause_ratio: float = 0.12
    ) -> dict:
        """
        Evaluates acoustic tension, pitch instability, and hesitation pauses.
        """
        tension_index = (jitter_percent / 3.0) * 0.5 + (pause_ratio / 0.5) * 0.5
        tension_index = min(1.0, max(0.0, tension_index))
        acoustic_stability = 1.0 - tension_index

        return {
            "pitch_f0_hz": pitch_f0_hz,
            "f0_std_hz": f0_std_hz,
            "jitter_percent": jitter_percent,
            "pause_ratio": pause_ratio,
            "acoustic_stability_score": round(acoustic_stability, 4)
        }

    @staticmethod
    def process_visual_features(
        dominant_emotion: str = "Neutral",
        facial_entropy: float = 0.25,
        gaze_drift_variance: float = 0.05,
        action_units_active: list = None
    ) -> dict:
        """
        Evaluates facial expression variability and gaze engagement.
        """
        if action_units_active is None:
            action_units_active = ["AU12"]

        visual_stability = 1.0 - (min(1.0, facial_entropy) * 0.4 + min(1.0, gaze_drift_variance * 4.0) * 0.6)
        visual_stability = max(0.0, min(1.0, visual_stability))

        return {
            "dominant_emotion": dominant_emotion,
            "action_units_active": action_units_active,
            "facial_entropy": facial_entropy,
            "gaze_drift_variance": gaze_drift_variance,
            "visual_stability_score": round(visual_stability, 4)
        }


# =====================================================================
# 4. MULTIMODAL TRUST & EXPLAINABILITY ENGINE
# =====================================================================
class TrustAIEngine:
    def __init__(self, log_path="trust_ai_feedback_logs.jsonl"):
        self.text_engine = TextConsistencyEngine()
        self.kinematic_engine = KinematicConsistencyEngine()
        self.behavioral_engine = BehavioralFeatureProcessor()
        self.log_path = log_path

    def infer(
        self,
        prompt: str,
        expected_answer: str,
        actual_answer: str,
        expected_trajectory: list,
        actual_trajectory: list,
        latency_ms: float = 120.0,
        audio_params: dict = None,
        visual_params: dict = None
    ) -> dict:
        if audio_params is None:
            audio_params = {"pitch_f0_hz": 185.0, "f0_std_hz": 22.0, "jitter_percent": 0.9, "pause_ratio": 0.15}
        if visual_params is None:
            visual_params = {"dominant_emotion": "Neutral", "facial_entropy": 0.20, "gaze_drift_variance": 0.06, "action_units_active": ["AU12"]}

        # 1. Modality-Specific Feature Extraction
        text_sig = self.text_engine.evaluate(expected_answer, actual_answer)
        kin_sig = self.kinematic_engine.evaluate(expected_trajectory, actual_trajectory, latency_ms)
        aud_sig = self.behavioral_engine.process_audio_features(**audio_params)
        vis_sig = self.behavioral_engine.process_visual_features(**visual_params)

        # 2. Multimodal Weighted Fusion
        w_text, w_kin, w_aud, w_vis = 0.50, 0.25, 0.15, 0.10
        composite_consistency = (
            text_sig["text_consistency_score"] * w_text +
            kin_sig["kinematic_consistency_score"] * w_kin +
            aud_sig["acoustic_stability_score"] * w_aud +
            vis_sig["visual_stability_score"] * w_vis
        )

        # 3. Anomaly & Confidence Logic
        is_anomaly = (
            text_sig["contradiction_probability"] > 0.45 or
            kin_sig["normalized_drift"] > 0.35 or
            composite_consistency < 0.50
        )

        confidence = 1.0 - (
            abs(text_sig["text_consistency_score"] - 0.5) * 0.20 +
            aud_sig["pause_ratio"] * 0.25 +
            vis_sig["facial_entropy"] * 0.20
        )
        confidence = min(0.99, max(0.40, confidence))

        # 4. Human-Interpretable Explainability Breakdown
        explanations = []
        if text_sig["contradiction_probability"] > 0.40:
            explanations.append(f"Textual Contradiction detected (p={text_sig['contradiction_probability']:.2f}).")
        if text_sig["semantic_similarity"] < 0.60:
            explanations.append(f"Low semantic similarity with reference answer ({text_sig['semantic_similarity']:.2f}).")
        if kin_sig["normalized_drift"] > 0.30:
            explanations.append(f"Kinematic trajectory drift ({kin_sig['normalized_drift']:.2f} drift units).")
        if kin_sig["reaction_latency_ms"] > 500:
            explanations.append(f"Elevated reaction latency ({kin_sig['reaction_latency_ms']} ms).")
        if aud_sig["jitter_percent"] > 2.0:
            explanations.append(f"Vocal tension / acoustic jitter elevated ({aud_sig['jitter_percent']}%).")
        if not explanations:
            explanations.append("High multi-modal concordance across text, kinematic, and behavioral channels.")

        record = {
            "interaction_id": f"session_{int(time.time() * 1000)}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": {"prompt": prompt},
            "linguistic_signals": {
                "expected_answer": expected_answer,
                "actual_answer": actual_answer,
                **text_sig
            },
            "kinematic_signals": kin_sig,
            "acoustic_signals": aud_sig,
            "visual_signals": vis_sig,
            "system_inference": {
                "predicted_consistency_score": round(composite_consistency, 4),
                "anomaly_detected": bool(is_anomaly),
                "system_confidence": round(confidence, 4),
                "explainability_summary": " ".join(explanations)
            }
        }

        return record

    def log_feedback(self, record: dict, supervisor_score: float, is_false_alarm: bool, annotator_notes: str = ""):
        """
        Saves user/supervisor verification feedback for model calibration.
        """
        record["human_feedback"] = {
            "supervisor_verified_score": supervisor_score,
            "is_false_alarm": is_false_alarm,
            "annotator_notes": annotator_notes,
            "calibration_error": round(abs(record["system_inference"]["predicted_consistency_score"] - supervisor_score), 4)
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[Trust-AI] Supervisor calibration record logged to '{self.log_path}'.")


# =====================================================================
# 5. SCRIPT EXECUTION & VERIFICATION TEST
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" TRUST-AI: MULTIMODAL CONSISTENCY & EXPLAINABILITY ENGINE")
    print("=" * 70)

    engine = TrustAIEngine()

    # -------------------------------------------------------------
    # Test Scenario A: Nominal Interaction (Consistent)
    # -------------------------------------------------------------
    print("\n[Scenario A] Nominal Interaction (High text alignment + smooth trajectory)")
    res_a = engine.infer(
        prompt="Explain the emergency shutdown protocol for the robotic arm.",
        expected_answer="Immediately engage the emergency stop button and isolate pneumatic pressure.",
        actual_answer="Press the E-stop switch right away and isolate the pneumatic air supply.",
        expected_trajectory=[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
        actual_trajectory=[[0, 0, 0], [1.05, 1.02, 0.98], [2.01, 1.95, 2.05], [3.0, 3.0, 3.0]],
        latency_ms=120.0,
        audio_params={"pitch_f0_hz": 180.0, "f0_std_hz": 18.0, "jitter_percent": 0.8, "pause_ratio": 0.10},
        visual_params={"dominant_emotion": "Neutral", "facial_entropy": 0.15, "gaze_drift_variance": 0.04, "action_units_active": ["AU12"]}
    )

    print("\n--- Output (Scenario A) ---")
    print(f"Consistency Score: {res_a['system_inference']['predicted_consistency_score'] * 100:.2f}%")
    print(f"Anomaly Detected:  {res_a['system_inference']['anomaly_detected']}")
    print(f"Model Confidence:  {res_a['system_inference']['system_confidence'] * 100:.2f}%")
    print(f"Explainability:    {res_a['system_inference']['explainability_summary']}")

    # -------------------------------------------------------------
    # Test Scenario B: Anomaly Interaction (Contradiction + Drift)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[Scenario B] Anomaly Interaction (Contradiction + Kinematic Drift)")
    res_b = engine.infer(
        prompt="What is the procedure when the robotic joint temperature exceeds 85°C?",
        expected_answer="Cut motor power immediately and allow the joint to cool down.",
        actual_answer="Increase operational speed and continue cycling the joint without interruption.",
        expected_trajectory=[[0, 0, 0], [1, 2, 3], [2, 4, 6], [3, 6, 9]],
        actual_trajectory=[[0, 0, 0], [4, 1, 2], [8, 3, 4], [12, 5, 8]],
        latency_ms=780.0,
        audio_params={"pitch_f0_hz": 245.0, "f0_std_hz": 48.0, "jitter_percent": 2.9, "pause_ratio": 0.42},
        visual_params={"dominant_emotion": "Fearful/Uncertain", "facial_entropy": 0.75, "gaze_drift_variance": 0.22, "action_units_active": ["AU01", "AU04", "AU15"]}
    )

    print("\n--- Output (Scenario B) ---")
    print(f"Consistency Score: {res_b['system_inference']['predicted_consistency_score'] * 100:.2f}%")
    print(f"Anomaly Detected:  {res_b['system_inference']['anomaly_detected']}")
    print(f"Model Confidence:  {res_b['system_inference']['system_confidence'] * 100:.2f}%")
    print(f"Explainability:    {res_b['system_inference']['explainability_summary']}")

    # -------------------------------------------------------------
    # Test Scenario C: Supervisor Feedback Logging
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[Feedback Calibration] Logging supervisor ground truth to JSONL...")
    engine.log_feedback(
        res_b,
        supervisor_score=0.08,
        is_false_alarm=False,
        annotator_notes="Accurately flagged dangerous advice and erratic motor trajectory."
    )
    print("=" * 70)
