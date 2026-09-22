"""
models/pretrained_face_model.py
======================================================================
Pretrained PyTorch Neural Network for Facial Action Units & Affect
Extracts a 64-dimensional affective embedding z_face and predicts:
- AU04 Brow Lowerer / Furrow (Stress indicator)
- AU12 Lip Corner Puller / Smile (Calm / Calibrated indicator)
- Binary Emotion Classification: 0: Smile (Calm), 1: Stressed
======================================================================
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

MODEL_SAVE_PATH = Path(__file__).resolve().parent / "pretrained_face_model.pt"

class FaceAffectNet(nn.Module):
    """
    Deep Facial Affect & Action Unit Feature Extractor.
    Encodes 6-dimensional facial geometric & blendshape cues into
    a 64-dimensional latent embedding z_face for multimodal fusion.
    """
    def __init__(self, input_dim: int = 6, embedding_dim: int = 64, num_classes: int = 4):
        super().__init__()
        
        # Non-linear feature representation encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.Tanh() # Normalized latent embedding [-1, 1]
        )
        
        # Regression head for FACS Action Units: AU04 (Furrow) & AU12 (Smile)
        self.au_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid() # Bound AU intensity in [0, 1]
        )
        
        # Classification head for 4 operational modes:
        # 0: Calm/Smile, 1: Stressed, 2: Surprised, 3: Frustrated
        self.classifier_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: [batch_size, input_dim]
        z_face = self.encoder(x) # [batch_size, 64]
        aus = self.au_head(z_face) # [batch_size, 2] -> [AU04, AU12]
        logits = self.classifier_head(z_face) # [batch_size, num_classes]
        return z_face, aus, logits

class FaceDataset(Dataset):
    LABEL_MAP = {
        "smile_calm": 0,
        "calm": 0,
        "smile": 0,
        "stressed": 1,
        "stress": 1,
        "surprised": 2,
        "surprise": 2,
        "frustrated": 3,
        "frustration": 3
    }

    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            data = json.load(f)
        
        self.x = []
        self.aus = []
        self.labels = []
        
        for item in data:
            self.x.append(item["mesh_features"])
            self.aus.append([item["au04_brow_furrow"], item["au12_smile"]])
            lbl = item.get("label", "smile_calm")
            self.labels.append(self.LABEL_MAP.get(lbl, 0))
            
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.aus = torch.tensor(self.aus, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.aus[idx], self.labels[idx]

def train_pretrained_face_model(data_path: str, epochs: int = 25, batch_size: int = 32, lr: float = 0.002):
    print("=" * 65)
    print(" TRAINING PRE-TRAINED FACIAL AFFECT & ACTION UNIT MODEL")
    print("=" * 65)
    
    dataset = FaceDataset(data_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    model = FaceAffectNet()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()
    
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_x, batch_au, batch_y in train_loader:
            optimizer.zero_grad()
            _, pred_aus, pred_logits = model(batch_x)
            loss_au = criterion_mse(pred_aus, batch_au)
            loss_cls = criterion_ce(pred_logits, batch_y)
            loss = 0.6 * loss_au + 0.4 * loss_cls
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_au, batch_y in val_loader:
                _, pred_aus, pred_logits = model(batch_x)
                loss_au = criterion_mse(pred_aus, batch_au)
                loss_cls = criterion_ce(pred_logits, batch_y)
                val_loss += (0.6 * loss_au + 0.4 * loss_cls).item()
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
            
    print(f"[Pretrained Face Model] Best weights saved -> {MODEL_SAVE_PATH}")
    return model

if __name__ == "__main__":
    json_data = Path(__file__).resolve().parent.parent / "data" / "facial_expression_dataset" / "facial_affect_samples.json"
    train_pretrained_face_model(str(json_data))
