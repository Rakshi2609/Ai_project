"""
models/multimodal_trust_fusion.py
======================================================================
Main Multimodal Deep Fusion Model for Predicting Human Trust in Cobots
Fuses feature representations extracted by pre-trained models:
- z_face in R^64 (from FaceAffectNet)
- z_voice in R^64 (from VoiceProsodyNet)
- z_robot in R^32 (from RobotBehaviorEncoder)
- z_physio in R^16 (from PhysiologicalEncoder)

Applies:
1. Dynamic Cross-Modal Softmax Attention with Noise Gating
2. Temporal LSTM State Dynamics (Rapid trust drop vs Gradual recovery)
3. Multi-task output: Continuous Trust Score T in [0.0, 1.0] and Control Policy
======================================================================
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

MODEL_SAVE_PATH = Path(__file__).resolve().parent / "trained_multimodal_trust_model.pt"

class CrossModalAttention(nn.Module):
    """
    Computes dynamic attention weights across modalities:
    [alpha_robot, alpha_face, alpha_voice, alpha_physio]
    """
    def __init__(self, common_dim: int = 32):
        super().__init__()
        self.proj_robot = nn.Linear(32, common_dim)
        self.proj_face = nn.Linear(64, common_dim)
        self.proj_voice = nn.Linear(64, common_dim)
        self.proj_physio = nn.Linear(16, common_dim)
        
        self.query = nn.Parameter(torch.randn(common_dim, 1))

    def forward(self, z_robot, z_face, z_voice, z_physio, snr_gate=None):
        # Project all modalities into common metric space
        h_robot = torch.tanh(self.proj_robot(z_robot)) # [B, D]
        h_face = torch.tanh(self.proj_face(z_face))
        h_voice = torch.tanh(self.proj_voice(z_voice))
        h_physio = torch.tanh(self.proj_physio(z_physio))
        
        # Stack modalities: [B, 4, D]
        stacked = torch.stack([h_robot, h_face, h_voice, h_physio], dim=1)
        
        # Compute attention scores: [B, 4, 1] -> [B, 4]
        scores = torch.matmul(stacked, self.query).squeeze(-1)
        
        # Apply SNR noise gating if environmental acoustics are poor
        if snr_gate is not None:
            # snr_gate: [B] in [0, 1] where 0 is noisy, 1 is clean
            # Downweight voice score index 2
            penalty = (1.0 - snr_gate) * 3.0
            scores[:, 2] = scores[:, 2] - penalty
            
        weights = F.softmax(scores, dim=-1) # [B, 4]
        
        # Weighted multimodal context representation: [B, D]
        # weights[:, i, None] * stacked[:, i, :]
        context = torch.sum(weights.unsqueeze(-1) * stacked, dim=1)
        
        return context, weights

class MultimodalTrustFusionModel(nn.Module):
    def __init__(self, lstm_hidden_dim: int = 64):
        super().__init__()
        
        # Pre-trained modality dimension projections
        self.physio_encoder = nn.Sequential(
            nn.Linear(3, 32), # [HR, HRV, EDA]
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Tanh()
        )
        
        # Cross-Modal Attention
        self.attention = CrossModalAttention(common_dim=32)
        
        # Temporal Bi-directional / Recurrent LSTM dynamics
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # Continuous Trust Score Head: T in [0.0, 1.0]
        self.trust_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid() # Continuous trust score bounded strictly in [0.0, 1.0]
        )
        
        # Multi-class Trust State Head (0: Under-Trust, 1: Calibrated, 2: Over-Trust)
        self.state_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
        
        # Mitigation Action Head (5 categories)
        self.action_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 5)
        )

    def forward(self, z_robot, z_face, z_voice, physio_raw, snr_gate=None, hidden_state=None):
        """
        Forward pass through multimodal attention fusion and temporal LSTM.
        Input shapes:
        - z_robot: [B, 32]
        - z_face: [B, 64]
        - z_voice: [B, 64]
        - physio_raw: [B, 3] -> [HR, HRV, EDA]
        """
        z_physio = self.physio_encoder(physio_raw) # [B, 16]
        
        # Cross-Modal Attention Fusion
        context, attn_weights = self.attention(z_robot, z_face, z_voice, z_physio, snr_gate) # [B, 32], [B, 4]
        
        # Add sequence dimension for LSTM: [B, 1, 32]
        seq_context = context.unsqueeze(1)
        
        if hidden_state is None:
            lstm_out, new_hidden = self.lstm(seq_context)
        else:
            lstm_out, new_hidden = self.lstm(seq_context, hidden_state)
            
        lstm_repr = lstm_out.squeeze(1) # [B, 64]
        
        # Multi-task predictions
        trust_score = self.trust_head(lstm_repr).squeeze(-1) # [B]
        state_logits = self.state_head(lstm_repr) # [B, 3]
        action_logits = self.action_head(lstm_repr) # [B, 5]
        
        return {
            "trust_score": trust_score,
            "state_logits": state_logits,
            "action_logits": action_logits,
            "attention_weights": attn_weights,
            "hidden_state": new_hidden
        }
