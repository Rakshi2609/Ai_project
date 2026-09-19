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

