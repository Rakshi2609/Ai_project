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
    Encodes 6-dimensional acoustic features (F0, Jitter, RMS, SpectralCentroid, ZCR, HNR)
    into a 64-dimensional latent embedding z_voice.
    Trained on RAVDESS (Zenodo) — real speech emotion recordings.
    """
    def __init__(self, input_dim: int = 6, embedding_dim: int = 64, num_classes: int = 4):
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
            nn.Tanh()   # Normalised latent acoustic embedding
        )

        # Tension & Jitter regression head
        self.tension_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()    # Bound in [0, 1]
        )

        # 4-Class Emotion classification head (Calm, Tense, Surprised, Subdued)
        self.classifier_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
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

        # Support both old synthetic schema and new real-data (RAVDESS) schema
        OLD_LABEL_MAP = {"calm": 0, "tremor_panic": 1, "warning_urgent": 2}
        NEW_LABEL_MAP = {"calm": 0, "tense": 1, "surprised": 2, "subdued": 3,
                         "happy": 0, "angry": 1, "fearful": 1, "neutral": 0}

        for item in data:
            # ── Feature vector ──────────────────────────────────────
            if "prosody_features" in item:
                vec = item["prosody_features"]          # new schema: 6-dim
            else:
                vec = item["prosody_vector"]            # old schema: 6-dim

            # ── Regression targets ──────────────────────────────────
            if "spectral_tension" in item:
                tension = float(item["spectral_tension"])
                jitter  = min(1.0, float(item["acoustic_jitter_pct"]) / 10.0)
            else:
                # Derive from prosody_features: [f0n, f0sn, rmsn, scn, zcrn, hnr]
                tension = float(vec[3]) if len(vec) > 3 else 0.3   # spectral centroid norm
                jitter  = float(vec[1]) if len(vec) > 1 else 0.15  # F0 std norm

            # ── Class label ─────────────────────────────────────────
            if "class_id" in item:
                lbl = int(item["class_id"])
            elif "emotion_label" in item:
                lbl = OLD_LABEL_MAP.get(item["emotion_label"], 0)
            else:
                lbl = NEW_LABEL_MAP.get(item.get("label", "calm"), 0)

            self.x.append(vec)
            self.targets.append([tension, jitter])
            self.labels.append(lbl)

        self.x       = torch.tensor(self.x,       dtype=torch.float32)
        self.targets = torch.tensor(self.targets,  dtype=torch.float32)
        self.labels  = torch.tensor(self.labels,   dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.targets[idx], self.labels[idx]

def train_pretrained_voice_model(data_path: str, epochs: int = 50, batch_size: int = 16, lr: float = 0.001):
    """
    Train VoiceProsodyNet on RAVDESS prosody features.

    Improvements over naive training:
    1. Class-weighted CrossEntropy  — handles calm=480 vs surprised=192 imbalance
    2. Gaussian noise augmentation  — regularises small 1,440-sample dataset
    3. Cosine annealing LR          — smooth convergence
    4. Larger epochs (50) + smaller batch (16) — more gradient updates per epoch
    """
    print("=" * 65)
    print(" TRAINING PRE-TRAINED VOCAL PROSODY & SPEECH EMOTION MODEL")
    print("=" * 65)
    
    dataset = VoiceDataset(data_path)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

    # ── Class weights: sqrt(inverse frequency) — gentler than full inverse  ──
    # Full inverse freq made model over-focus on 192-sample classes → accuracy drop
    # sqrt smooths the imbalance: calm(0.75→0.87), surprised(1.875→1.37)
    all_labels = dataset.labels.numpy()
    from collections import Counter
    import math
    counts = Counter(all_labels.tolist())
    n_total = len(all_labels)
    n_classes = 4
    weights = torch.tensor(
        [math.sqrt(n_total / (n_classes * max(counts.get(c, 1), 1))) for c in range(n_classes)],
        dtype=torch.float32
    )

    model     = VoiceProsodyNet()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    criterion_mse = nn.MSELoss()
    criterion_ce  = nn.CrossEntropyLoss(weight=weights)   # ← class-weighted

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_x, batch_tgt, batch_y in train_loader:
            optimizer.zero_grad()

            # Gaussian noise augmentation — improves generalisation on small dataset
            noise = torch.randn_like(batch_x) * 0.02
            batch_x_aug = batch_x + noise

            _, pred_tgt, pred_logits = model(batch_x_aug)
            loss_regr = criterion_mse(pred_tgt, batch_tgt)
            loss_cls  = criterion_ce(pred_logits, batch_y)
            loss = 0.4 * loss_regr + 0.6 * loss_cls   # weight CE higher
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, batch_tgt, batch_y in val_loader:
                _, pred_tgt, pred_logits = model(batch_x)
                loss_regr = criterion_mse(pred_tgt, batch_tgt)
                loss_cls  = criterion_ce(pred_logits, batch_y)
                val_loss += (0.4 * loss_regr + 0.6 * loss_cls).item()
                correct  += (torch.argmax(pred_logits, dim=1) == batch_y).sum().item()
                total    += batch_y.size(0)

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)
        accuracy       = (correct / total) * 100.0

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.1f}%")

        # Save best model by validation accuracy (not loss — avoids overfit saves)
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"[Pretrained Voice Model] Best Val Acc: {best_val_acc:.1f}% | "
          f"Saved -> {MODEL_SAVE_PATH}")
    return model


if __name__ == "__main__":
    json_data = Path(__file__).resolve().parent.parent / "data" / "voice_emotion_dataset" / "speech_prosody_samples.json"
    train_pretrained_voice_model(str(json_data))
