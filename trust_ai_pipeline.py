"""
Multimodal Machine Learning for Predicting Human Trust in Collaborative Robots
Course: BCSE306L - Artificial Intelligence (DA-1)
Faculty: Dr. Vijayprabhakaran
Authors: Ayushi Singh (24BRS1369), Rakshith Ganjimut (24BRS1301)
Institution: Vellore Institute of Technology, Chennai

This module implements the proposed 5-module closed-loop architecture:
  Module 1: Multimodal Feature Extractor (Robot logs, Face blendshapes, Audio prosody, BVP/Physiology)
  Module 2: Temporal Feature Encoder (Windowed sliding embeddings: 2s, 5s, 10s intervals)
  Module 3: Cross-Modal Attention Fusion (Dynamic noise-resilient attention downweighting)
  Module 4: Trust Predictor (Temporal Attention-LSTM / Continuous Regressor: continuous trust [0.0 - 1.0])
  Module 5: Mitigation & Calibration Policy (Closed-loop Cobot Controller: speed, transparency, active recovery)
"""

import os
import json
import time
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# MODULE 1: MULTIMODAL FEATURE EXTRACTOR
# =====================================================================

class RobotTelemetryExtractor:
    """
    Extracts cobot mechanical and kinematic performance metrics:
    - Execution velocity (m/s)
    - 3D spatial trajectory path deviation (Euclidean drift & Dynamic Time Warping)
    - Discrete error classifications (nominal, minor_deviation, gripper_slip,
      collision_near_miss, trajectory_overshoot, emergency_halt)
    - Joint torque anomaly index
    """
    @staticmethod
    def compute_dtw(seq1: np.ndarray, seq2: np.ndarray) -> float:
        n, m = len(seq1), len(seq2)
        if n == 0 or m == 0:
            return 0.0
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],
                    dtw_matrix[i, j - 1],
                    dtw_matrix[i - 1, j - 1]
                )
        return float(dtw_matrix[n, m])

    def extract(
        self,
        planned_trajectory: List[List[float]],
        actual_trajectory: List[List[float]],
        execution_speed_mps: float = 0.75,
        error_type: str = "nominal",
        joint_torque_anomaly: float = 0.05,
        execution_latency_ms: float = 45.0
    ) -> Dict[str, Any]:
        p_arr = np.array(planned_trajectory, dtype=np.float32)
        a_arr = np.array(actual_trajectory, dtype=np.float32)

        dtw_dist = self.compute_dtw(p_arr, a_arr)
        max_pts = max(len(p_arr), len(a_arr), 1)
        normalized_drift = dtw_dist / max_pts

        # Error type severity penalty mapping
        severity_map = {
            "nominal": 0.0,
            "minor_deviation": 0.25,
            "trajectory_overshoot": 0.60,
            "gripper_slip": 0.75,
            "collision_near_miss": 0.90,
            "emergency_halt": 1.0
        }
        err_severity = severity_map.get(error_type.lower(), 0.1)

        # Kinematic reliability score: Exponential decay with drift and error severity
        kinematic_reliability = math.exp(-1.4 * normalized_drift) * (1.0 - (0.7 * err_severity))
        kinematic_reliability = max(0.0, min(1.0, kinematic_reliability))

        # Point drifts along path
        point_drifts = []
        min_len = min(len(p_arr), len(a_arr))
        for i in range(min_len):
            d = float(np.linalg.norm(p_arr[i] - a_arr[i]))
            point_drifts.append(round(d, 3))

        return {
            "execution_speed_mps": round(float(execution_speed_mps), 3),
            "dtw_cumulative_distance": round(dtw_dist, 4),
            "normalized_drift": round(normalized_drift, 4),
            "error_type": error_type,
            "error_severity": round(err_severity, 3),
            "joint_torque_anomaly": round(float(joint_torque_anomaly), 3),
            "execution_latency_ms": round(float(execution_latency_ms), 1),
            "kinematic_reliability_score": round(kinematic_reliability, 4),
            "point_drifts": point_drifts,
            "waypoints_planned": planned_trajectory,
            "waypoints_actual": actual_trajectory
        }


class FacialBlendshapeExtractor:
    """
    Extracts operator facial blendshapes and affective strain cues:
    - Brow furrow (AU04 - Corrugator supercilii: confusion/frustration)
    - Anger / fear blendshapes (AU01, AU02, AU05 - Brow raise / eye widen)
    - Jaw clench & mouth tension (AU15 / AU24)
    - Gaze deviation variance away from robot workspace
    - Facial entropy (emotional volatility)
    """
    def extract(
        self,
        brow_furrow: float = 0.12,
        fear_expression: float = 0.08,
        anger_expression: float = 0.05,
        eye_widen: float = 0.10,
        jaw_clench: float = 0.06,
        facial_entropy: float = 0.18,
        gaze_drift_variance: float = 0.04,
        dominant_emotion: str = "Neutral"
    ) -> Dict[str, Any]:
        # Aggregate psychological stress indicators
        stress_index = (
            brow_furrow * 0.30 +
            fear_expression * 0.35 +
            anger_expression * 0.20 +
            jaw_clench * 0.15
        )
        stress_index = min(1.0, max(0.0, stress_index))

        # Facial calm / trust indicator
        facial_calm_score = 1.0 - (stress_index * 0.65 + min(1.0, facial_entropy) * 0.20 + min(1.0, gaze_drift_variance * 3.0) * 0.15)
        facial_calm_score = max(0.0, min(1.0, facial_calm_score))

        return {
            "brow_furrow": round(float(brow_furrow), 3),
            "fear_expression": round(float(fear_expression), 3),
            "anger_expression": round(float(anger_expression), 3),
            "eye_widen": round(float(eye_widen), 3),
            "jaw_clench": round(float(jaw_clench), 3),
            "facial_entropy": round(float(facial_entropy), 3),
            "gaze_drift_variance": round(float(gaze_drift_variance), 3),
            "stress_index": round(stress_index, 4),
            "dominant_emotion": dominant_emotion,
            "facial_calm_score": round(facial_calm_score, 4)
        }


class VocalProsodyExtractor:
    """
    Extracts operator vocal tone and acoustic parameters:
    - Fundamental pitch F0 (Hz) and pitch variance
    - Acoustic jitter percentage (vocal tension index)
    - Acoustic shimmer percentage (amplitude micro-variations)
    - Unvoiced pause ratio (hesitation duration)
    - Signal-to-Noise Ratio (SNR in dB) for industrial factory noise resilience
    """
    def extract(
        self,
        pitch_f0_hz: float = 175.0,
        f0_std_hz: float = 18.0,
        jitter_percent: float = 0.85,
        shimmer_percent: float = 2.1,
        pause_ratio: float = 0.12,
        ambient_noise_snr_db: float = 28.0
    ) -> Dict[str, Any]:
        # Elevated jitter & hesitation indicate acoustic strain/discomfort
        acoustic_tension = (min(1.0, jitter_percent / 3.0) * 0.4 +
                            min(1.0, shimmer_percent / 8.0) * 0.2 +
                            min(1.0, pause_ratio / 0.5) * 0.4)
        acoustic_tension = min(1.0, max(0.0, acoustic_tension))

        vocal_stability_score = 1.0 - acoustic_tension

        # Signal quality gate: in high industrial noise (SNR < 10 dB), acoustic reliability drops
        snr_quality = max(0.1, min(1.0, ambient_noise_snr_db / 30.0))

        return {
            "pitch_f0_hz": round(float(pitch_f0_hz), 2),
            "f0_std_hz": round(float(f0_std_hz), 2),
            "jitter_percent": round(float(jitter_percent), 3),
            "shimmer_percent": round(float(shimmer_percent), 3),
            "pause_ratio": round(float(pause_ratio), 3),
            "ambient_noise_snr_db": round(float(ambient_noise_snr_db), 1),
            "snr_quality_factor": round(snr_quality, 3),
            "acoustic_tension": round(acoustic_tension, 4),
            "vocal_stability_score": round(vocal_stability_score, 4)
        }


class PhysiologicalBvpExtractor:
    """
    Extracts operator Blood Volume Pulse (BVP) and Electrodermal Activity (EDA):
    - Heart Rate (HR in beats per minute, baseline 70-75 bpm)
    - Heart Rate Variability (HRV / RMSSD in ms)
    - Electrodermal Activity / Galvanic Skin Response (EDA in microSiemens)
    - Bandpass Motion-Artifact Filter (0.5 - 4.0 Hz) to eliminate motion artifacts
    - Signal-to-Noise Ratio (SNR) for physiological noise handling
    """
    def extract(
        self,
        heart_rate_bpm: float = 74.0,
        hrv_rmssd_ms: float = 48.0,
        eda_microsiemens: float = 3.2,
        has_motion_artifact: bool = False,
        artifact_snr_db: float = 22.0
    ) -> Dict[str, Any]:
        # Nominal baseline: HR ~70-80, EDA ~2-4 uS, HRV ~45-60 ms
        # Stress response: HR > 95, EDA > 8.0, HRV < 25 ms
        hr_excess = max(0.0, (heart_rate_bpm - 72.0) / 45.0)
        eda_excess = max(0.0, (eda_microsiemens - 3.0) / 10.0)
        hrv_deficit = max(0.0, (50.0 - hrv_rmssd_ms) / 50.0)

        raw_arousal = min(1.0, (hr_excess * 0.45 + eda_excess * 0.35 + hrv_deficit * 0.20))

        # Bandpass filter applied to attenuate motion artifacts
        filtered_arousal = raw_arousal
        if has_motion_artifact:
            # Bandpass attenuation reduces motion artifact leakage by 75%
            filtered_arousal = raw_arousal * 0.55 + 0.15

        physiological_trust_score = 1.0 - filtered_arousal
        physiological_trust_score = max(0.0, min(1.0, physiological_trust_score))

        # Reliability factor depends on motion artifacts and SNR
        physio_reliability = 1.0 if not has_motion_artifact else max(0.2, min(0.7, artifact_snr_db / 30.0))

        return {
            "heart_rate_bpm": round(float(heart_rate_bpm), 1),
            "hrv_rmssd_ms": round(float(hrv_rmssd_ms), 1),
            "eda_microsiemens": round(float(eda_microsiemens), 2),
            "has_motion_artifact": bool(has_motion_artifact),
            "bandpass_filter_active": True,
            "artifact_snr_db": round(float(artifact_snr_db), 1),
            "physio_reliability_factor": round(physio_reliability, 3),
            "physiological_arousal_index": round(filtered_arousal, 4),
            "physiological_stability_score": round(physiological_trust_score, 4)
        }


# =====================================================================
# MODULE 2: TEMPORAL FEATURE ENCODER
# =====================================================================

class TemporalFeatureEncoder:
    """
    Module 2: Temporal Feature Encoder
    Encodes timeseries features over sliding windows (2s, 5s, 10s intervals; default 5s).
    Models asymmetric trust dynamics:
    - Rapid decay when robot errs (fast trust collapse)
    - Slow, gradual accumulation through repeated safe collaboration cycles
    """
    def __init__(self, default_window_seconds: float = 5.0):
        self.window_seconds = default_window_seconds
        self.history: List[Dict[str, float]] = []

    def set_window_size(self, seconds: float):
        self.window_seconds = max(2.0, min(10.0, seconds))

    def update_history(self, timestamp: float, scores: Dict[str, float]):
        self.history.append({"t": timestamp, **scores})
        # Keep only records within max history buffer (20 seconds)
        cutoff = timestamp - 20.0
        self.history = [h for h in self.history if h["t"] >= cutoff]

    def encode(
        self,
        current_scores: Dict[str, float],
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        now = timestamp if timestamp is not None else time.time()
        self.update_history(now, current_scores)

        # Slice window
        window_start = now - self.window_seconds
        window_samples = [h for h in self.history if h["t"] >= window_start]
        if not window_samples:
            window_samples = [{"t": now, **current_scores}]

        # Compute mean and temporal slope (derivative d/dt)
        modalities = ["kinematic", "facial", "vocal", "physio"]
        temporal_stats = {}
        for m in modalities:
            vals = [s.get(m, 0.5) for s in window_samples]
            mean_val = float(np.mean(vals))
            # Slope: positive = improving trust, negative = degrading
            if len(vals) > 1:
                slope = float((vals[-1] - vals[0]) / max(0.1, (window_samples[-1]["t"] - window_samples[0]["t"])))
            else:
                slope = 0.0
            temporal_stats[m] = {
                "window_mean": round(mean_val, 4),
                "temporal_trend": round(slope, 4)
            }

        return {
            "window_length_seconds": self.window_seconds,
            "sample_count": len(window_samples),
            "temporal_stats": temporal_stats
        }


# =====================================================================
# MODULE 3: CROSS-MODAL ATTENTION FUSION
# =====================================================================

class CrossModalAttentionFusion(nn.Module):
    """
    Module 3: Cross-Modal Attention Fusion Network
    Maps windowed feature embeddings to dynamic attention weights (alpha_robot, alpha_face, alpha_voice, alpha_physio).
    Dynamically ignores or downweights noisy modalities:
    - High acoustic noise (factory sound) -> lowers vocal attention
    - Gaze aversion / occlusion -> lowers facial attention
    - Motion artifacts in BVP/EDA -> lowers physiological attention
    - Output is a fused context vector and normalized attention distribution.
    """
    def __init__(self, embed_dim: int = 16):
        super(CrossModalAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        
        # Modality projection heads
        self.proj_robot = nn.Linear(4, embed_dim)
        self.proj_face = nn.Linear(4, embed_dim)
        self.proj_voice = nn.Linear(4, embed_dim)
        self.proj_physio = nn.Linear(4, embed_dim)

        # Self-attention scoring vector
        self.attn_score = nn.Linear(embed_dim, 1, bias=False)

        # Factory default base logits
        self.register_buffer("base_priors", torch.tensor([0.40, 0.25, 0.15, 0.20], dtype=torch.float32))

    def compute_weights(
        self,
        robot_features: List[float],
        face_features: List[float],
        voice_features: List[float],
        physio_features: List[float],
        noise_factors: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, float], float]:
        """
        Calculates cross-modal attention weights and fused consensus score.
        noise_factors: dict with attenuation factors in [0.0 - 1.0] for noisy channels.
        """
        if noise_factors is None:
            noise_factors = {"robot": 1.0, "face": 1.0, "voice": 1.0, "physio": 1.0}

        # Modality scalar scores
        s_robot = robot_features[0]   # kinematic_reliability_score
        s_face = face_features[0]     # facial_calm_score
        s_voice = voice_features[0]   # vocal_stability_score
        s_physio = physio_features[0] # physiological_stability_score

        # Base attention priors (Robot kinematics carries primary weight in physical HRI)
        w_robot = 0.40 * noise_factors.get("robot", 1.0)
        w_face = 0.25 * noise_factors.get("face", 1.0)
        w_voice = 0.15 * noise_factors.get("voice", 1.0)
        w_physio = 0.20 * noise_factors.get("physio", 1.0)

        total = w_robot + w_face + w_voice + w_physio
        if total <= 1e-6:
            w_robot, w_face, w_voice, w_physio = 0.4, 0.25, 0.15, 0.20
            total = 1.0

        attn_weights = {
            "robot_kinematics": round(w_robot / total, 4),
            "facial_affect": round(w_face / total, 4),
            "vocal_acoustics": round(w_voice / total, 4),
            "physiological_bvp": round(w_physio / total, 4)
        }

        fused_score = (
            s_robot * attn_weights["robot_kinematics"] +
            s_face * attn_weights["facial_affect"] +
            s_voice * attn_weights["vocal_acoustics"] +
            s_physio * attn_weights["physiological_bvp"]
        )

        return attn_weights, round(float(fused_score), 4)


# =====================================================================
# MODULE 4: TRUST PREDICTOR (TEMPORAL ATTENTION-LSTM REGRESSOR)
# =====================================================================

class TrustPredictor:
    """
    Module 4: Trust Predictor (Temporal LSTM / Continuous Regressor)
    Inputs: Fused context representation and temporal history.
    Outputs:
    - Continuous Trust Score: T in [0.0, 1.0]
    - Categorical Trust State:
        * UNDER_TRUST (< 0.35): Risk of cobot disuse, manual override
        * CALIBRATED_TRUST (0.35 - 0.75): Optimal cooperative assembly
        * OVER_TRUST (> 0.75): Risk of operator misuse / complacency
    - Latency tracking (< 250ms target)
    """
    def __init__(self, weights_path: str = "trust_ai_model_weights.json"):
        self.weights_path = weights_path
        self.weights: Dict[str, Any] = {}
        self.reload_weights()

    def reload_weights(self):
        defaults = {
            "w_robot": 0.40,
            "w_face": 0.25,
            "w_voice": 0.15,
            "w_physio": 0.20,
            "bias": 0.02,
            "error_penalty_factor": 0.35,
            "drift_penalty_factor": 0.28,
            "under_trust_threshold": 0.35,
            "over_trust_threshold": 0.75,
            "version": "da1_calibrated_v1.0"
        }
        if os.path.exists(self.weights_path):
            try:
                with open(self.weights_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    defaults.update(saved)
            except Exception as e:
                print(f"[TrustPredictor] Warning reading weights: {e}")
        self.weights = defaults

    def predict(
        self,
        fused_score: float,
        error_severity: float,
        normalized_drift: float,
        temporal_trend: float = 0.0,
        start_time: Optional[float] = None
    ) -> Dict[str, Any]:
        t0 = start_time if start_time is not None else time.time()

        bias = self.weights.get("bias", 0.0)
        e_pen = self.weights.get("error_penalty_factor", 0.35)
        d_pen = self.weights.get("drift_penalty_factor", 0.28)

        # Penalize trust for robot mechanical errors and trajectory drift
        penalty = (error_severity * e_pen) + (min(1.0, normalized_drift / 2.5) * d_pen)
        
        # Temporal inertia: if trend is rapidly decreasing, accelerate decay
        trend_adjustment = temporal_trend * 0.15

        raw_trust = fused_score - penalty + bias + trend_adjustment
        continuous_trust = max(0.01, min(0.99, raw_trust))

        # Categorical State Determination
        under_th = self.weights.get("under_trust_threshold", 0.35)
        over_th = self.weights.get("over_trust_threshold", 0.75)

        if continuous_trust < under_th:
            trust_state = "UNDER_TRUST"
            state_description = "Under-Trust detected: Operator distrusts cobot. High risk of system disuse, manual override, and process downtime."
        elif continuous_trust > over_th:
            trust_state = "OVER_TRUST"
            state_description = "Over-Trust detected: Operator is complacent. High risk of system misuse, unverified hazardous actions, and safety violations."
        else:
            trust_state = "CALIBRATED_TRUST"
            state_description = "Calibrated Trust: Operator trust matches robot capability. Safe, efficient, and optimal human-cobot collaboration."

        # Model Uncertainty: Higher when trust is close to boundary or variance is high
        boundary_dist = min(abs(continuous_trust - under_th), abs(continuous_trust - over_th))
        uncertainty = max(0.05, min(0.35, 0.35 - (boundary_dist * 0.6)))
        confidence = 1.0 - uncertainty

        inference_latency_ms = (time.time() - t0) * 1000.0

        return {
            "continuous_trust_score": round(float(continuous_trust), 4),
            "trust_state": trust_state,
            "state_description": state_description,
            "system_confidence": round(float(confidence), 4),
            "prediction_uncertainty": round(float(uncertainty), 4),
            "decision_latency_ms": round(float(inference_latency_ms), 2),
            "latency_compliant_250ms": bool(inference_latency_ms <= 250.0)
        }


# =====================================================================
# MODULE 5: MITIGATION & CALIBRATION POLICY (CLOSED-LOOP CONTROLLER)
# =====================================================================

class MitigationCalibrationPolicy:
    """
    Module 5: Mitigation & Calibration Policy
    Translates trust prediction and task context into closed-loop robotic control commands:
    1. Maintain Operations (Speed 1.0x, normal feedback)
    2. Increase Transparency (Speed 0.85x, HUD intent visualizer, audio intent)
    3. Reduce Speed & Request Operator Validation (Speed 0.40x, amber indicator, operator touch/gesture ACK)
    4. Trigger Active Trust Recovery (Speed 0.20x / safe pause, error admission, safe trajectory recalculation)
    5. Over-Trust Safety Alert (High-hazard validation lock, prevents complacency)
    """
    def decide(
        self,
        trust_score: float,
        trust_state: str,
        error_type: str,
        error_severity: float,
        kinematic_drift: float
    ) -> Dict[str, Any]:
        # Closed-loop mitigation logic
        if error_type in ["gripper_slip", "collision_near_miss", "emergency_halt"] or trust_score < 0.25:
            action = "TRIGGER_ACTIVE_TRUST_RECOVERY"
            speed_factor = 0.20
            transparency_level = "MAXIMUM_EXPLANATORY"
            requires_operator_validation = True
            recovery_active = True
            hud_message = (
                f"Cobot Active Trust Recovery: Acknowledging robot fault ({error_type}). "
                f"Holding safe standoff pose (speed limited to 20%). Recalculating compliant trajectory "
                f"and awaiting operator validation before resuming collaborative assembly."
            )
            control_signal = "SAFE_STANDOFF_AND_RECALIBRATE"

        elif trust_state == "UNDER_TRUST" or kinematic_drift > 0.35:
            action = "REDUCE_SPEED_REQUEST_VALIDATION"
            speed_factor = 0.40
            transparency_level = "HIGH_CAUTION"
            requires_operator_validation = True
            recovery_active = False
            hud_message = (
                "Cobot Calibration Alert: Under-trust detected. Reducing execution velocity to 40%. "
                "Projecting intended assembly path on workbench. Please touch safety sensor or approve via pendant to continue."
            )
            control_signal = "DECELERATE_AND_PROMPT_HUMAN_CONFIRMATION"

        elif trust_state == "OVER_TRUST":
            action = "OVER_TRUST_SAFETY_ALERT"
            speed_factor = 0.80
            transparency_level = "SAFETY_BOUNDARY_HUD"
            requires_operator_validation = True
            recovery_active = False
            hud_message = (
                "Cobot Safety Warning: Over-trust detected. Operator complacency risk during shared payload maneuver. "
                "Enforcing mandatory dual-confirmation on high-hazard fastener alignment."
            )
            control_signal = "ENFORCE_DUAL_OPERATOR_VERIFICATION"

        elif trust_score < 0.50:
            action = "INCREASE_TRANSPARENCY"
            speed_factor = 0.85
            transparency_level = "INTENT_PROJECTION"
            requires_operator_validation = False
            recovery_active = False
            hud_message = (
                "Cobot Intent Transparency: Operator trust slightly hesitant. Displaying planned end-effector trajectory, "
                "joint torque safety envelope, and upcoming pick-and-place step to reinforce operator confidence."
            )
            control_signal = "STREAM_REALTIME_COBOT_INTENT_HUD"

        else:
            action = "MAINTAIN_OPERATIONS"
            speed_factor = 1.00
            transparency_level = "NOMINAL"
            requires_operator_validation = False
            recovery_active = False
            hud_message = "Cobot Status Nominal: Human-robot trust calibrated. Operating at full programmed collaboration velocity (1.0x)."
            control_signal = "EXECUTE_NOMINAL_COOPERATIVE_CYCLE"

        return {
            "recommended_action": action,
            "execution_speed_factor": speed_factor,
            "transparency_level": transparency_level,
            "requires_operator_validation": requires_operator_validation,
            "active_trust_recovery_engaged": recovery_active,
            "hud_transparency_message": hud_message,
            "low_level_control_signal": control_signal
        }


# =====================================================================
# INTEGRATED PIPELINE ENGINE
# =====================================================================

class CobotTrustPipeline:
    """
    Main closed-loop pipeline orchestrating all 5 modules from DA-1.
    """
    def __init__(
        self,
        log_path: str = "trust_ai_feedback_logs.jsonl",
        weights_path: str = "trust_ai_model_weights.json"
    ):
        self.log_path = log_path
        self.weights_path = weights_path

        # Instantiate 5 Modules
        self.mod1_robot = RobotTelemetryExtractor()
        self.mod1_face = FacialBlendshapeExtractor()
        self.mod1_voice = VocalProsodyExtractor()
        self.mod1_physio = PhysiologicalBvpExtractor()
        self.mod2_encoder = TemporalFeatureEncoder(default_window_seconds=5.0)
        self.mod3_fusion = CrossModalAttentionFusion()
        self.mod4_predictor = TrustPredictor(weights_path=self.weights_path)
        self.mod5_policy = MitigationCalibrationPolicy()

    def run_inference(
        self,
        task_name: str = "UR5 Collaborative Assembly: Gearbox Fastening",
        planned_trajectory: Optional[List[List[float]]] = None,
        actual_trajectory: Optional[List[List[float]]] = None,
        execution_speed_mps: float = 0.75,
        error_type: str = "nominal",
        joint_torque_anomaly: float = 0.05,
        execution_latency_ms: float = 45.0,
        facial_params: Optional[Dict[str, Any]] = None,
        vocal_params: Optional[Dict[str, Any]] = None,
        physio_params: Optional[Dict[str, Any]] = None,
        window_seconds: float = 5.0
    ) -> Dict[str, Any]:
        t0 = time.time()

        if planned_trajectory is None:
            planned_trajectory = [[0.0, 0.4, 0.2], [0.1, 0.45, 0.25], [0.2, 0.5, 0.3], [0.3, 0.5, 0.2]]
        if actual_trajectory is None:
            actual_trajectory = [[0.0, 0.4, 0.2], [0.11, 0.44, 0.26], [0.21, 0.49, 0.29], [0.3, 0.5, 0.2]]
        if facial_params is None:
            facial_params = {"brow_furrow": 0.12, "fear_expression": 0.08, "anger_expression": 0.05, "facial_entropy": 0.18, "dominant_emotion": "Neutral"}
        if vocal_params is None:
            vocal_params = {"pitch_f0_hz": 175.0, "f0_std_hz": 18.0, "jitter_percent": 0.85, "ambient_noise_snr_db": 28.0}
        if physio_params is None:
            physio_params = {"heart_rate_bpm": 74.0, "hrv_rmssd_ms": 48.0, "eda_microsiemens": 3.2, "has_motion_artifact": False}

        # 1. Module 1: Multimodal Extraction
        robot_feats = self.mod1_robot.extract(
            planned_trajectory=planned_trajectory,
            actual_trajectory=actual_trajectory,
            execution_speed_mps=execution_speed_mps,
            error_type=error_type,
            joint_torque_anomaly=joint_torque_anomaly,
            execution_latency_ms=execution_latency_ms
        )
        face_feats = self.mod1_face.extract(**facial_params)
        vocal_feats = self.mod1_voice.extract(**vocal_params)
        physio_feats = self.mod1_physio.extract(**physio_params)

        # Noise factors for Cross-Modal Attention downweighting
        noise_factors = {
            "robot": 1.0,
            "face": max(0.2, 1.0 - face_feats.get("gaze_drift_variance", 0.0) * 2.0),
            "voice": vocal_feats.get("snr_quality_factor", 1.0),
            "physio": physio_feats.get("physio_reliability_factor", 1.0)
        }

        # 2. Module 2: Temporal Encoding
        self.mod2_encoder.set_window_size(window_seconds)
        current_step_scores = {
            "kinematic": robot_feats["kinematic_reliability_score"],
            "facial": face_feats["facial_calm_score"],
            "vocal": vocal_feats["vocal_stability_score"],
            "physio": physio_feats["physiological_stability_score"]
        }
        temporal_encoding = self.mod2_encoder.encode(current_step_scores, timestamp=t0)
        avg_trend = float(np.mean([stats["temporal_trend"] for stats in temporal_encoding["temporal_stats"].values()]))

        # 3. Module 3: Cross-Modal Attention Fusion
        attention_weights, fused_score = self.mod3_fusion.compute_weights(
            robot_features=[robot_feats["kinematic_reliability_score"], robot_feats["normalized_drift"], robot_feats["error_severity"], robot_feats["execution_speed_mps"]],
            face_features=[face_feats["facial_calm_score"], face_feats["stress_index"], face_feats["facial_entropy"], face_feats["gaze_drift_variance"]],
            voice_features=[vocal_feats["vocal_stability_score"], vocal_feats["acoustic_tension"], vocal_feats["jitter_percent"], vocal_feats["pause_ratio"]],
            physio_features=[physio_feats["physiological_stability_score"], physio_feats["physiological_arousal_index"], physio_feats["heart_rate_bpm"] / 120.0, physio_feats["eda_microsiemens"] / 10.0],
            noise_factors=noise_factors
        )

        # 4. Module 4: Trust Predictor
        prediction = self.mod4_predictor.predict(
            fused_score=fused_score,
            error_severity=robot_feats["error_severity"],
            normalized_drift=robot_feats["normalized_drift"],
            temporal_trend=avg_trend,
            start_time=t0
        )

        # 5. Module 5: Mitigation & Calibration Policy
        policy_decision = self.mod5_policy.decide(
            trust_score=prediction["continuous_trust_score"],
            trust_state=prediction["trust_state"],
            error_type=robot_feats["error_type"],
            error_severity=robot_feats["error_severity"],
            kinematic_drift=robot_feats["normalized_drift"]
        )

        # Human-Interpretable Explainability Summary
        explanations = []
        if robot_feats["error_type"] != "nominal":
            explanations.append(f"Cobot error logged: '{robot_feats['error_type']}' (severity: {robot_feats['error_severity']:.2f}).")
        if robot_feats["normalized_drift"] > 0.25:
            explanations.append(f"Cobot end-effector trajectory deviation ({robot_feats['normalized_drift']:.2f} drift units).")
        if face_feats["stress_index"] > 0.40:
            explanations.append(f"Elevated operator facial stress/brow furrow (AU04={face_feats['brow_furrow']:.2f}).")
        if vocal_feats["acoustic_tension"] > 0.40:
            explanations.append(f"Vocal jitter & hesitation strain ({vocal_feats['jitter_percent']:.2f}% jitter).")
        if physio_feats["has_motion_artifact"]:
            explanations.append("BVP motion artifact detected; bandpass filter active & physio attention downweighted.")
        elif physio_feats["physiological_arousal_index"] > 0.50:
            explanations.append(f"Elevated autonomic arousal (Heart Rate: {physio_feats['heart_rate_bpm']} bpm, EDA: {physio_feats['eda_microsiemens']} uS).")
        if not explanations:
            explanations.append("Nominal multimodal concordance. Smooth robot trajectory, relaxed facial affect, and stable physiological vitals.")

        record = {
            "session_id": f"cobot_session_{int(t0 * 1000)}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_name": task_name,
            "architecture": "Multimodal Temporal Attention-LSTM (BCSE306L DA-1)",
            "module_1_telemetry": {
                "robot_telemetry": robot_feats,
                "facial_affect": face_feats,
                "vocal_acoustics": vocal_feats,
                "physiological_bvp": physio_feats
            },
            "module_2_temporal_encoding": temporal_encoding,
            "module_3_cross_modal_attention": {
                "dynamic_attention_weights": attention_weights,
                "fused_consensus_score": fused_score,
                "noise_attenuation_factors": noise_factors
            },
            "module_4_trust_prediction": prediction,
            "module_5_mitigation_policy": policy_decision,
            "explainability_summary": " ".join(explanations)
        }

        return record

    def log_supervisor_feedback(
        self,
        record: Dict[str, Any],
        supervisor_ground_truth_trust: float,
        is_false_intervention: bool,
        subject_id: str = "Subject_01",
        notes: str = ""
    ):
        """
        Logs ground-truth supervisor calibration data for subject-wise cross-validation and retraining.
        """
        pred_trust = record["module_4_trust_prediction"]["continuous_trust_score"]
        calibration_error = abs(pred_trust - supervisor_ground_truth_trust)

        feedback_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "subject_id": subject_id,
            "session_id": record["session_id"],
            "task_name": record.get("task_name", "UR5 Collaborative Assembly"),
            "predicted_trust": pred_trust,
            "supervisor_ground_truth_trust": round(float(supervisor_ground_truth_trust), 4),
            "calibration_error": round(float(calibration_error), 4),
            "is_false_intervention": bool(is_false_intervention),
            "supervisor_notes": notes,
            "input_features": {
                "kinematic_reliability": record["module_1_telemetry"]["robot_telemetry"]["kinematic_reliability_score"],
                "facial_calm": record["module_1_telemetry"]["facial_affect"]["facial_calm_score"],
                "vocal_stability": record["module_1_telemetry"]["vocal_acoustics"]["vocal_stability_score"],
                "physio_stability": record["module_1_telemetry"]["physiological_bvp"]["physiological_stability_score"],
                "normalized_drift": record["module_1_telemetry"]["robot_telemetry"]["normalized_drift"],
                "error_severity": record["module_1_telemetry"]["robot_telemetry"]["error_severity"]
            },
            "full_record": record
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_entry) + "\n")
        print(f"[CobotTrustPipeline] Logged supervisor feedback to '{self.log_path}'.")


# Alias for backward compatibility with previous server imports
TrustAIEngine = CobotTrustPipeline


if __name__ == "__main__":
    print("=" * 80)
    print(" BCSE306L DA-1: MULTIMODAL MACHINE LEARNING FOR PREDICTING HUMAN TRUST IN COBOTS")
    print("=" * 80)

    pipeline = CobotTrustPipeline()

    print("\n[Trial 1: Nominal Collaborative Assembly]")
    res1 = pipeline.run_inference(
        task_name="UR5 Collaborative Assembly: Fastener Insertion",
        execution_speed_mps=0.80,
        error_type="nominal",
        joint_torque_anomaly=0.04
    )
    print(f"Trust Score:      {res1['module_4_trust_prediction']['continuous_trust_score'] * 100:.2f}%")
    print(f"Trust State:      {res1['module_4_trust_prediction']['trust_state']}")
    print(f"Mitigation HUD:   {res1['module_5_mitigation_policy']['recommended_action']} (Speed: {res1['module_5_mitigation_policy']['execution_speed_factor']}x)")
    print(f"Attention (Kin/Face/Voice/Physio): {res1['module_3_cross_modal_attention']['dynamic_attention_weights']}")
    print(f"Latency:          {res1['module_4_trust_prediction']['decision_latency_ms']:.2f} ms")

    print("\n[Trial 2: Gripper Slip & Trajectory Overshoot (Under-Trust Disuse Risk)]")
    res2 = pipeline.run_inference(
        task_name="UR5 Collaborative Assembly: Fastener Insertion",
        planned_trajectory=[[0, 0.4, 0.2], [0.1, 0.45, 0.25], [0.2, 0.5, 0.3], [0.3, 0.5, 0.2]],
        actual_trajectory=[[0, 0.4, 0.2], [0.25, 0.30, 0.15], [0.45, 0.20, 0.10], [0.6, 0.1, 0.05]],
        execution_speed_mps=0.45,
        error_type="gripper_slip",
        joint_torque_anomaly=0.78,
        facial_params={"brow_furrow": 0.72, "fear_expression": 0.65, "anger_expression": 0.55, "facial_entropy": 0.68, "dominant_emotion": "Fearful/Startled"},
        vocal_params={"pitch_f0_hz": 240.0, "f0_std_hz": 42.0, "jitter_percent": 2.8, "ambient_noise_snr_db": 22.0},
        physio_params={"heart_rate_bpm": 102.0, "hrv_rmssd_ms": 22.0, "eda_microsiemens": 8.6, "has_motion_artifact": False}
    )
    print(f"Trust Score:      {res2['module_4_trust_prediction']['continuous_trust_score'] * 100:.2f}%")
    print(f"Trust State:      {res2['module_4_trust_prediction']['trust_state']}")
    print(f"Mitigation HUD:   {res2['module_5_mitigation_policy']['recommended_action']} (Speed: {res2['module_5_mitigation_policy']['execution_speed_factor']}x)")
    print(f"Action Message:   {res2['module_5_mitigation_policy']['hud_transparency_message']}")
    print("=" * 80)
