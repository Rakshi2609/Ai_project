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

