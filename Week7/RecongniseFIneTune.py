import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix)
import pandas as pd
import random

# ========================
# CONFIGURATION
# ========================
VIDEO_DIRS = {
    "shoplifter": r"D:\Cellula_tech_intern\Week7\Shop DataSet\Shop DataSet\shop lifters",
    "non_shoplifter": r"D:\Cellula_tech_intern\Week7\Shop DataSet\Shop DataSet\non shop lifters"
}
FRAME_SIZE = (224, 224)   # resize for CNN
FRAMES_PER_VIDEO = 16     # sample this many frames per video
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42
TEST_SPLIT = 0.1
VAL_SPLIT = 0.2
MODEL_SAVE_PATH = "shoplifter_model_finetuned.pth"
METRICS_CSV = "metrics_test.csv"
PLOTS_DIR = "plots"
FINE_TUNE = True   # <-- set to False for feature extraction

os.makedirs(PLOTS_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ========================
# DATASET
# ========================
class VideoDataset(Dataset):
    def __init__(self, video_dirs, transform=None, frames_per_video=16):
        self.samples = []
        self.transform = transform
        self.frames_per_video = frames_per_video

        for label_name, folder in video_dirs.items():
            label = 1 if label_name == "shoplifter" else 0
            for file in os.listdir(folder):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    self.samples.append((os.path.join(folder, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._load_video_frames(video_path)
        frames = self._ensure_frame_count(frames)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        frames = torch.stack(frames)  # (frames, C, H, W)
        return frames, torch.tensor(label, dtype=torch.float32)

    def _load_video_frames(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idxs = self._sample_frame_indices(total_frames)

        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_idxs:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, FRAME_SIZE)
                frames.append(frame)
        cap.release()
        return frames

    def _sample_frame_indices(self, total_frames):
        if total_frames <= 0:
            return []
        if total_frames <= self.frames_per_video:
            return list(range(total_frames))
        step = total_frames / self.frames_per_video
        return [int(i * step) for i in range(self.frames_per_video)]

    def _ensure_frame_count(self, frames):
        if len(frames) == 0:
            black = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
            return [black.copy() for _ in range(self.frames_per_video)]
        if len(frames) >= self.frames_per_video:
            return frames[:self.frames_per_video]
        while len(frames) < self.frames_per_video:
            frames.append(frames[-1].copy())
        return frames

# ========================
# MODEL (pretrained ResNet + LSTM)
# ========================
class CNNLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_classes=1, fine_tune=True):
        super(CNNLSTM, self).__init__()
        # Load pretrained ResNet18
        base_model = models.resnet18(pretrained=True)

        # Freeze params if only feature extracting
        if not fine_tune:
            for param in base_model.parameters():
                param.requires_grad = False

        # Remove final FC layer
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # (B,512,1,1)

        # LSTM + classifier
        self.lstm = nn.LSTM(512, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        b, f, c, h, w = x.size()
        x = x.view(b * f, c, h, w)
        feats = self.feature_extractor(x)           # (b*f, 512, 1, 1)
        feats = feats.view(b, f, -1)                # (b, f, 512)
        lstm_out, _ = self.lstm(feats)              # (b, f, hidden_size)
        out = self.fc(lstm_out[:, -1, :])           # (b, num_classes)
        return out.squeeze(1)

# ========================
# EVALUATION
# ========================
def evaluate_model(model, data_loader):
    model.eval()
    all_labels, all_probs, all_preds, losses = [], [], [], []
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for frames, labels in data_loader:
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)
            logits = model(frames)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            losses.append(loss.item())

    metrics = {}
    y_true, y_pred, y_prob = np.array(all_labels), np.array(all_preds), np.array(all_probs)
    metrics['loss'] = float(np.mean(losses)) if losses else 0.0
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    metrics['precision'], metrics['recall'], metrics['f1'] = float(p), float(r), float(f1)
    try:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics['roc_auc'] = float('nan')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    metrics.update(dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)))
    return metrics

# ========================
# TRAINING
# ========================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()
        running_loss, all_labels, all_preds = 0.0, [], []
        for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(frames)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        val_metrics = evaluate_model(model, val_loader)
        val_loss, val_acc = val_metrics['loss'], val_metrics['accuracy']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}  TrainLoss={train_loss:.4f}  TrainAcc={train_acc:.4f}  "
              f"ValLoss={val_loss:.4f}  ValAcc={val_acc:.4f}")
    return history

# ========================
# MAIN
# ========================
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = VideoDataset(VIDEO_DIRS, transform=transform, frames_per_video=FRAMES_PER_VIDEO)
    n = len(full_dataset)
    if n == 0:
        raise RuntimeError("No videos found. Check VIDEO_DIRS paths.")

    test_size, val_size = int(n * TEST_SPLIT), int(n * VAL_SPLIT)
    train_size = n - val_size - test_size
    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size],
                                                generator=torch.Generator().manual_seed(RANDOM_SEED))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = CNNLSTM(fine_tune=FINE_TUNE).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()

    if FINE_TUNE:
        optimizer = optim.Adam([
            {"params": model.feature_extractor.parameters(), "lr": LR * 0.1},
            {"params": model.lstm.parameters(), "lr": LR},
            {"params": model.fc.parameters(), "lr": LR}
        ])
    else:
        optimizer = optim.Adam([
            {"params": model.lstm.parameters(), "lr": LR},
            {"params": model.fc.parameters(), "lr": LR}
        ])

    history = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Training complete. Model saved as {MODEL_SAVE_PATH}")

    # Plots
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(); plt.plot(epochs_range, history['train_loss'], label='train_loss')
    plt.plot(epochs_range, history['val_loss'], label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss Curve')
    plt.savefig(os.path.join(PLOTS_DIR, 'loss_curve.png')); plt.close()

    plt.figure(); plt.plot(epochs_range, history['train_acc'], label='train_acc')
    plt.plot(epochs_range, history['val_acc'], label='val_acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy Curve')
    plt.savefig(os.path.join(PLOTS_DIR, 'accuracy_curve.png')); plt.close()

    print(f"✅ Plots saved to: {PLOTS_DIR}")

    # Test evaluation
    test_metrics = evaluate_model(model, test_loader)
    pd.DataFrame([test_metrics]).to_csv(METRICS_CSV, index=False)
    print(f"✅ Test metrics saved to {METRICS_CSV}")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    # Confusion matrix
    all_labels, all_preds = [], []
    model.eval()
    with torch.no_grad():
        for frames, labels in test_loader:
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)
            preds = (torch.sigmoid(model(frames)) > 0.5).float()
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    cm = confusion_matrix(all_labels, all_preds, labels=[0,1])
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion matrix'); plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['non_shoplifter','shoplifter'], rotation=45)
    plt.yticks(tick_marks, ['non_shoplifter','shoplifter'])
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'), ha="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label'); plt.xlabel('Predicted label'); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png')); plt.close()

    print(f"✅ Confusion matrix saved to {os.path.join(PLOTS_DIR, 'confusion_matrix.png')}")
