"""
models/pretrained_voice_model.py
======================================================================
Pretrained PyTorch Neural Network for Speech Emotion & Vocal Prosody
Extracts a 64-dimensional acoustic embedding z_voice and predicts:
- Acoustic Tension Index [0.0, 1.0]
- Pitch Jitter % (vocal micro-tremor indicator)
- 3-Class Vocal Affect: 0: Calm, 1: Tremor/Panic, 2: Warning/Urgent
======================================================================
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

MODEL_SAVE_PATH = Path(__file__).resolve().parent / "pretrained_voice_model.pt"

class VoiceProsodyNet(nn.Module):
    """
    Deep Vocal Prosody & Emotion Feature Extractor.
    Encodes 5-dimensional acoustic features (Pitch, Jitter, Intensity, Tension, SNR)
    into a 64-dimensional latent embedding z_voice.
    """
    def __init__(self, input_dim: int = 5, embedding_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.Tanh() # Normalized latent acoustic embedding
        )
        
        # Tension & Jitter regression head
        self.tension_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid() # Bound in [0, 1]
        )
        
        # 3-Class Emotion classification head (Calm, Panic, Warning)
        self.classifier_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        # x: [batch_size, input_dim]
        z_voice = self.encoder(x)
        tension_pred = self.tension_head(z_voice) # [tension, norm_jitter]
        logits = self.classifier_head(z_voice)
        return z_voice, tension_pred, logits

class VoiceDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            data = json.load(f)
            
        self.x = []
        self.targets = []
        self.labels = []
        
        label_map = {"calm": 0, "tremor_panic": 1, "warning_urgent": 2}
        
        for item in data:
            self.x.append(item["prosody_vector"])
            # Normalized targets: [spectral_tension, acoustic_jitter_pct / 10.0]
            self.targets.append([item["spectral_tension"], min(1.0, item["acoustic_jitter_pct"] / 10.0)])
            self.labels.append(label_map[item["emotion_label"]])
            
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.targets[idx], self.labels[idx]

def train_pretrained_voice_model(data_path: str, epochs: int = 25, batch_size: int = 32, lr: float = 0.002):
    print("=" * 65)
    print(" TRAINING PRE-TRAINED VOCAL PROSODY & SPEECH EMOTION MODEL")
    print("=" * 65)
    
    dataset = VoiceDataset(data_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    model = VoiceProsodyNet()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()
    
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_x, batch_tgt, batch_y in train_loader:
            optimizer.zero_grad()
            _, pred_tgt, pred_logits = model(batch_x)
            loss_regr = criterion_mse(pred_tgt, batch_tgt)
            loss_cls = criterion_ce(pred_logits, batch_y)
            loss = 0.5 * loss_regr + 0.5 * loss_cls
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_tgt, batch_y in val_loader:
                _, pred_tgt, pred_logits = model(batch_x)
                loss_regr = criterion_mse(pred_tgt, batch_tgt)
                loss_cls = criterion_ce(pred_logits, batch_y)
                val_loss += (0.5 * loss_regr + 0.5 * loss_cls).item()
                preds = torch.argmax(pred_logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
                
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = (correct / total) * 100.0
        
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.1f}%")
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
    print(f"[Pretrained Voice Model] Best weights saved -> {MODEL_SAVE_PATH}")
    return model

if __name__ == "__main__":
    json_data = Path(__file__).resolve().parent.parent / "data" / "voice_emotion_dataset" / "speech_prosody_samples.json"
    train_pretrained_voice_model(str(json_data))
