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


