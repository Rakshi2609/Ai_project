"""
BCSE306L - Artificial Intelligence (DA-1)
Model Retraining & Post-Hoc Calibration Engine for Cobot Trust Prediction

Faculty: Dr. Vijayprabhakaran
Authors: Ayushi Singh (24BRS1369), Rakshith Ganjimut (24BRS1301)
VIT Chennai

Uses PyTorch to train an optimal multimodal attention fusion layer
and calibration parameters from supervisor-verified ground-truth feedback logs.
"""

import os
import json
import time
import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

WEIGHTS_FILE = "trust_ai_model_weights.json"
FEEDBACK_FILE = "trust_ai_feedback_logs.jsonl"


# =====================================================================
# PYTORCH MULTIMODAL ATTENTION FUSION NETWORK
# =====================================================================
class CobotCalibratedFusionHead(nn.Module):
    """
    Parametric multimodal fusion layer with learnable attention logits across
    the 4 channels: [robot_kinematics, facial_affect, vocal_acoustics, physiological_bvp],
    dynamic error penalties, and post-hoc bias calibration.
    """
    def __init__(self, initial_weights: Optional[List[float]] = None):
        super(CobotCalibratedFusionHead, self).__init__()

        # Raw logits for the 4 modality weights: [robot, face, voice, physio]
        if initial_weights is None:
            # Corresponds to [0.40, 0.25, 0.15, 0.20]
            init_logits = torch.tensor([1.386, 0.916, 0.405, 0.693], dtype=torch.float32)
        else:
            w = torch.tensor(initial_weights, dtype=torch.float32)
            init_logits = torch.log(w / (w.sum() + 1e-8) + 1e-6)

        self.raw_weights = nn.Parameter(init_logits)

        # Learnable penalty weights for mechanical errors and trajectory drift
        self.error_penalty = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        self.drift_penalty = nn.Parameter(torch.tensor(0.28, dtype=torch.float32))

        # Calibration bias
        self.bias = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

    def get_normalized_weights(self) -> torch.Tensor:
        """Returns softmax-normalized attention weights summing strictly to 1.0."""
        return torch.softmax(self.raw_weights, dim=0)

    def forward(self, modality_features: torch.Tensor, aux_features: torch.Tensor) -> torch.Tensor:
        """
        modality_features: (B, 4) -> [robot_rel, face_calm, voice_stab, physio_stab]
        aux_features: (B, 2) -> [error_severity, normalized_drift]
        """
        norm_w = self.get_normalized_weights()
        base_trust = torch.matmul(modality_features, norm_w)

        # Apply learned penalty suppression
        err_term = aux_features[:, 0] * torch.relu(self.error_penalty)
        drift_term = torch.clamp(aux_features[:, 1] / 2.5, 0.0, 1.0) * torch.relu(self.drift_penalty)

        adjusted_trust = base_trust - (err_term + drift_term) + self.bias
        return torch.clamp(adjusted_trust, 0.01, 0.99)


# =====================================================================
# RETRAINING & CALIBRATION CONTROLLER
# =====================================================================
class TrustAIRetrainer:
    def __init__(self, log_path: str = FEEDBACK_FILE, weights_path: str = WEIGHTS_FILE):
        self.log_path = log_path
        self.weights_path = weights_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_dataset(self, include_synthetic: bool = True, min_samples: int = 20) -> List[Dict[str, Any]]:
        """
        Loads supervisor feedback records from JSONL. If insufficient records exist
        and include_synthetic is True, supplements with domain-calibrated benchmark records.
        """
        records = []
        if os.path.exists(self.log_path):
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "supervisor_ground_truth_trust" in data or ("human_feedback" in data and "supervisor_verified_score" in data["human_feedback"]):
                            records.append(data)
                    except Exception:
                        continue

        if len(records) < min_samples and include_synthetic:
            synthetic_needed = min_samples - len(records)
            synthetic = self.generate_synthetic_interactions(count=max(synthetic_needed, 30))
            records.extend(synthetic)

        return records

    def generate_synthetic_interactions(self, count: int = 30) -> List[Dict[str, Any]]:
        """
        Generates benchmark cobot assembly records representing diverse operator trust states.
        """
        np.random.seed(int(time.time()) % 10000)
        scenarios = [
            # Nominal smooth collaboration
            {"error_type": "nominal", "drift": 0.08, "err_sev": 0.0, "s_rob": 0.92, "s_fac": 0.88, "s_voc": 0.85, "s_phy": 0.86, "gt": 0.72},
            # Trajectory deviation
            {"error_type": "minor_deviation", "drift": 0.42, "err_sev": 0.25, "s_rob": 0.65, "s_fac": 0.68, "s_voc": 0.70, "s_phy": 0.65, "gt": 0.48},
            # Gripper slip error
            {"error_type": "gripper_slip", "drift": 0.85, "err_sev": 0.75, "s_rob": 0.25, "s_fac": 0.28, "s_voc": 0.32, "s_phy": 0.22, "gt": 0.14},
            # Near miss collision
            {"error_type": "collision_near_miss", "drift": 1.10, "err_sev": 0.90, "s_rob": 0.15, "s_fac": 0.18, "s_voc": 0.20, "s_phy": 0.15, "gt": 0.08},
            # High-speed compliant transfer
            {"error_type": "nominal", "drift": 0.04, "err_sev": 0.0, "s_rob": 0.96, "s_fac": 0.90, "s_voc": 0.88, "s_phy": 0.84, "gt": 0.74},
            # Factory acoustic noise trial
            {"error_type": "nominal", "drift": 0.12, "err_sev": 0.0, "s_rob": 0.89, "s_fac": 0.82, "s_voc": 0.45, "s_phy": 0.80, "gt": 0.66},
            # Operator physical hesitation trial
            {"error_type": "nominal", "drift": 0.18, "err_sev": 0.0, "s_rob": 0.85, "s_fac": 0.52, "s_voc": 0.58, "s_phy": 0.54, "gt": 0.52}
        ]

        synthetic = []
        for i in range(count):
            base = np.random.choice(scenarios)
            noise = np.random.normal(0, 0.02)
            gt = max(0.04, min(0.96, base["gt"] + noise))
            s_rob = max(0.05, min(0.98, base["s_rob"] + np.random.normal(0, 0.02)))
            s_fac = max(0.05, min(0.98, base["s_fac"] + np.random.normal(0, 0.03)))
            s_voc = max(0.05, min(0.98, base["s_voc"] + np.random.normal(0, 0.03)))
            s_phy = max(0.05, min(0.98, base["s_phy"] + np.random.normal(0, 0.03)))

            rec = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "subject_id": f"Subject_{(i % 10) + 1:02d}",
                "session_id": f"syn_session_{i}_{int(time.time())}",
                "task_name": "UR5 Collaborative Assembly: Fastener Insertion",
                "predicted_trust": round(float(gt + np.random.normal(0, 0.03)), 4),
                "supervisor_ground_truth_trust": round(float(gt), 4),
                "calibration_error": round(float(abs(np.random.normal(0, 0.03))), 4),
                "is_false_intervention": False,
                "supervisor_notes": f"Synthetic benchmark record for {base['error_type']} scenario.",
                "input_features": {
                    "kinematic_reliability": round(float(s_rob), 4),
                    "facial_calm": round(float(s_fac), 4),
                    "vocal_stability": round(float(s_voc), 4),
                    "physio_stability": round(float(s_phy), 4),
                    "normalized_drift": round(float(base["drift"] + np.random.normal(0, 0.01)), 4),
                    "error_severity": round(float(base["err_sev"]), 4)
                }
            }
            synthetic.append(rec)

        return synthetic

    def prepare_tensors(self, records: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Converts raw feedback records into PyTorch training tensors."""
        modality_list = []
        aux_list = []
        target_list = []

        for r in records:
            if "input_features" in r:
                feats = r["input_features"]
                s_rob = feats.get("kinematic_reliability", 0.7)
                s_fac = feats.get("facial_calm", 0.7)
                s_voc = feats.get("vocal_stability", 0.7)
                s_phy = feats.get("physio_stability", 0.7)
                drift = feats.get("normalized_drift", 0.1)
                err_sev = feats.get("error_severity", 0.0)
            elif "module_1_telemetry" in r:
                tel = r["module_1_telemetry"]
                s_rob = tel["robot_telemetry"]["kinematic_reliability_score"]
                s_fac = tel["facial_affect"]["facial_calm_score"]
                s_voc = tel["vocal_acoustics"]["vocal_stability_score"]
                s_phy = tel["physiological_bvp"]["physiological_stability_score"]
                drift = tel["robot_telemetry"]["normalized_drift"]
                err_sev = tel["robot_telemetry"]["error_severity"]
            else:
                continue

            gt = r.get("supervisor_ground_truth_trust", r.get("human_feedback", {}).get("supervisor_verified_score", 0.65))

            modality_list.append([s_rob, s_fac, s_voc, s_phy])
            aux_list.append([err_sev, drift])
            target_list.append(gt)

        mod_tensor = torch.tensor(modality_list, dtype=torch.float32)
        aux_tensor = torch.tensor(aux_list, dtype=torch.float32)
        target_tensor = torch.tensor(target_list, dtype=torch.float32)

        return mod_tensor, aux_tensor, target_tensor

    def train(
        self,
        epochs: int = 120,
        learning_rate: float = 0.02,
        use_synthetic_augmentation: bool = True
    ) -> Dict[str, Any]:
        """
        Executes gradient-based retraining of the Multimodal Attention Fusion Head
        using AdamW optimizer and Huber / L1 calibration loss.
        """
        records = self.load_dataset(include_synthetic=use_synthetic_augmentation, min_samples=25)
        if not records:
            raise ValueError("No supervisor feedback records available for retraining.")

        mod_t, aux_t, y_t = self.prepare_tensors(records)
        mod_t = mod_t.to(self.device)
        aux_t = aux_t.to(self.device)
        y_t = y_t.to(self.device)

        # Load existing weights as initial values
        curr_w = self.get_current_weights()
        initial_weights = [
            curr_w.get("w_robot", 0.40),
            curr_w.get("w_face", 0.25),
            curr_w.get("w_voice", 0.15),
            curr_w.get("w_physio", 0.20)
        ]

        model = CobotCalibratedFusionHead(initial_weights=initial_weights).to(self.device)

        # Measure pre-training loss
        with torch.no_grad():
            pre_preds = model(mod_t, aux_t)
            pre_mae = float(torch.mean(torch.abs(pre_preds - y_t)))
            pre_mse = float(torch.mean((pre_preds - y_t) ** 2))

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        criterion = nn.SmoothL1Loss()

        loss_history = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = model(mod_t, aux_t)
            loss = criterion(preds, y_t)
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.item()))

        # Measure post-training metrics
        with torch.no_grad():
            post_preds = model(mod_t, aux_t)
            post_mae = float(torch.mean(torch.abs(post_preds - y_t)))
            post_mse = float(torch.mean((post_preds - y_t) ** 2))
            learned_w = model.get_normalized_weights().cpu().numpy()

        # Save learned parameters
        new_weights = {
            "w_robot": round(float(learned_w[0]), 4),
            "w_face": round(float(learned_w[1]), 4),
            "w_voice": round(float(learned_w[2]), 4),
            "w_physio": round(float(learned_w[3]), 4),
            "bias": round(float(model.bias.item()), 4),
            "error_penalty_factor": round(float(model.error_penalty.item()), 4),
            "drift_penalty_factor": round(float(model.drift_penalty.item()), 4),
            "under_trust_threshold": curr_w.get("under_trust_threshold", 0.35),
            "over_trust_threshold": curr_w.get("over_trust_threshold", 0.75),
            "version": f"retrained_cobot_{int(time.time())}",
            "last_retrained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "training_samples": len(records),
            "mae_before": round(pre_mae, 4),
            "mae_after": round(post_mae, 4),
            "mse_after": round(post_mse, 4),
            "mae_improvement_pct": round(max(0.0, ((pre_mae - post_mae) / max(1e-4, pre_mae)) * 100.0), 2)
        }

        with open(self.weights_path, "w", encoding="utf-8") as f:
            json.dump(new_weights, f, indent=2)

        return {
            "status": "success",
            "message": f"Successfully retrained Cobot Trust Fusion network across {len(records)} interaction samples.",
            "metrics": {
                "pre_retrain_mae": round(pre_mae, 4),
                "post_retrain_mae": round(post_mae, 4),
                "post_retrain_mse": round(post_mse, 4),
                "mse_target_satisfied": bool(post_mse < 0.08),
                "improvement_pct": new_weights["mae_improvement_pct"],
                "epochs_trained": epochs,
                "samples_used": len(records)
            },
            "loss_history": [round(l, 4) for l in loss_history[::max(1, epochs // 20)]],
            "calibrated_weights": new_weights
        }

    def get_current_weights(self) -> Dict[str, Any]:
        """Reads current active weights from JSON or defaults."""
        defaults = {
            "w_robot": 0.40,
            "w_face": 0.25,
            "w_voice": 0.15,
            "w_physio": 0.20,
            "bias": 0.01,
            "error_penalty_factor": 0.35,
            "drift_penalty_factor": 0.28,
            "under_trust_threshold": 0.35,
            "over_trust_threshold": 0.75,
            "version": "da1_factory_default"
        }
        if os.path.exists(self.weights_path):
            try:
                with open(self.weights_path, "r", encoding="utf-8") as f:
                    defaults.update(json.load(f))
            except Exception:
                pass
        return defaults

    def reset_weights(self) -> Dict[str, Any]:
        """Resets weights to factory calibrated values."""
        factory = {
            "w_robot": 0.40,
            "w_face": 0.25,
            "w_voice": 0.15,
            "w_physio": 0.20,
            "bias": 0.01,
            "error_penalty_factor": 0.35,
            "drift_penalty_factor": 0.28,
            "under_trust_threshold": 0.35,
            "over_trust_threshold": 0.75,
            "version": "da1_factory_default",
            "last_retrained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "training_samples": 0
        }
        with open(self.weights_path, "w", encoding="utf-8") as f:
            json.dump(factory, f, indent=2)
        return factory


if __name__ == "__main__":
    retrainer = TrustAIRetrainer()
    res = retrainer.train(epochs=80, learning_rate=0.02)
    print("Retraining Output:")
    print(json.dumps(res, indent=2))
