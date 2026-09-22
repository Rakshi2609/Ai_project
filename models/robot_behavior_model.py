"""
models/robot_behavior_model.py
======================================================================
PyTorch Robot Kinematics & Behavior Feature Encoder
Encodes UR5 collaborative manipulator state:
- TCP execution speed (m/s)
- 3D Euclidean trajectory path deviation (mm)
- Joint torque anomaly index [0.0, 1.0]
- Error severity [0.0, 1.0]
Outputs a 32-dimensional robot latent embedding z_robot.
======================================================================
"""

import torch
import torch.nn as nn

class RobotBehaviorEncoder(nn.Module):
    def __init__(self, input_dim: int = 4, embedding_dim: int = 32):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.Tanh() # Normalized kinematic embedding
        )
        
        # Auxiliary head to predict mechanical reliability score [0.0, 1.0]
        self.reliability_head = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch_size, 4] -> [speed, deviation_mm, torque_anomaly, error_severity]
        z_robot = self.net(x)
        rel = self.reliability_head(z_robot)
        return z_robot, rel
