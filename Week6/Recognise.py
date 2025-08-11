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
    "shoplifter": r"D:\Cellula_tech_intern\Week6\Shop DataSet\Shop DataSet\shop lifters",
    "non_shoplifter": r"D:\Cellula_tech_intern\Week6\Shop DataSet\Shop DataSet\non shop lifters"
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
MODEL_SAVE_PATH = "shoplifter_model_from_scratch.pth"
METRICS_CSV = "metrics_test.csv"
PLOTS_DIR = "plots"

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
        frames = self._load_video_frames(video_path)  # list of HxWxC np arrays
        # ensure we have exactly frames_per_video frames (pad or sample)
        frames = self._ensure_frame_count(frames)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]  # list of (C,H,W) tensors

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
        # If video shorter than desired, just take all available frames
        if total_frames <= 0:
            return []
        if total_frames <= self.frames_per_video:
            return list(range(total_frames))
        # evenly spaced sampling
        step = total_frames / self.frames_per_video
        return [int(i * step) for i in range(self.frames_per_video)]

    def _ensure_frame_count(self, frames):
        # If zero frames (corrupt video), create frames_per_video black frames
        if len(frames) == 0:
            black = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
            return [black.copy() for _ in range(self.frames_per_video)]

        # If too many (shouldn't happen) -> crop
        if len(frames) >= self.frames_per_video:
            return frames[:self.frames_per_video]

        # If fewer, repeat last frame until desired length
        while len(frames) < self.frames_per_video:
            frames.append(frames[-1].copy())
        return frames

# ========================
# MODEL (from scratch)
# ========================
class CNNLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_classes=1):
        super(CNNLSTM, self).__init__()
        # Use ResNet18 architecture but DO NOT load pretrained weights -> train from scratch
        base_model = models.resnet18(pretrained=False)
        # Remove the final fully connected layer and the avgpool is kept to produce a 512-dim vector
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # outputs (B,512,1,1)
        self.lstm = nn.LSTM(512, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # logits output
        # no sigmoid here; we'll use BCEWithLogitsLoss for numerical stability

    def forward(self, x):
        # x: (batch, frames, C, H, W)
        b, f, c, h, w = x.size()
        x = x.view(b * f, c, h, w)                  # (b*f, C, H, W)
        feats = self.feature_extractor(x)           # (b*f, 512, 1, 1)
        feats = feats.view(b, f, -1)                # (b, f, 512)
        lstm_out, _ = self.lstm(feats)              # (b, f, hidden_size)
        out = self.fc(lstm_out[:, -1, :])           # (b, num_classes) -> logits
        return out.squeeze(1)                       # (b,) logits

# ========================
# TRAIN / VAL / TEST LOOPS
# ========================
def evaluate_model(model, data_loader):
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
    losses = []
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for frames, labels in data_loader:
            frames = frames.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(frames)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            losses.append(loss.item())

    metrics = {}
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics['loss'] = float(np.mean(losses)) if len(losses) > 0 else 0.0
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    metrics['precision'] = float(p)
    metrics['recall'] = float(r)
    metrics['f1'] = float(f1)
    # roc auc requires both classes present
    try:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics['roc_auc'] = float('nan')

    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    metrics['tn'] = int(tn); metrics['fp'] = int(fp); metrics['fn'] = int(fn); metrics['tp'] = int(tp)

    return metrics

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []

        for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            frames = frames.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(frames)  # logits shape (b,)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)

        # Validation
        val_metrics = evaluate_model(model, val_loader)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}  TrainLoss={train_loss:.4f}  TrainAcc={train_acc:.4f}  ValLoss={val_loss:.4f}  ValAcc={val_acc:.4f}")

    return history

# ========================
# MAIN SCRIPT
# ========================
if __name__ == "__main__":
    # Transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Build dataset and splits
    full_dataset = VideoDataset(VIDEO_DIRS, transform=transform, frames_per_video=FRAMES_PER_VIDEO)
    n = len(full_dataset)
    if n == 0:
        raise RuntimeError("No videos found. Check VIDEO_DIRS paths.")

    test_size = int(n * TEST_SPLIT)
    val_size = int(n * VAL_SPLIT)
    train_size = n - val_size - test_size
    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size],
                                                generator=torch.Generator().manual_seed(RANDOM_SEED))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = CNNLSTM().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Train
    history = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)

    # Save model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Training complete. Model saved as {MODEL_SAVE_PATH}")

    # Plot training curves
    epochs_range = range(1, len(history['train_loss']) + 1)

    plt.figure()
    plt.plot(epochs_range, history['train_loss'], label='train_loss')
    plt.plot(epochs_range, history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig(os.path.join(PLOTS_DIR, 'loss_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(epochs_range, history['train_acc'], label='train_acc')
    plt.plot(epochs_range, history['val_acc'], label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.savefig(os.path.join(PLOTS_DIR, 'accuracy_curve.png'))
    plt.close()

    print(f"✅ Plots saved to: {PLOTS_DIR}")

    # Evaluate on test set and save metrics
    test_metrics = evaluate_model(model, test_loader)
    df_metrics = pd.DataFrame([test_metrics])
    df_metrics.to_csv(METRICS_CSV, index=False)
    print(f"✅ Test metrics saved to {METRICS_CSV}")
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    # Additionally save confusion matrix as plot
    # Compute preds and labels again (for confusion matrix visualization)
    all_labels = []
    all_preds = []
    model.eval()
    with torch.no_grad():
        for frames, labels in test_loader:
            frames = frames.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(frames)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    cm = confusion_matrix(all_labels, all_preds, labels=[0,1])
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['non_shoplifter','shoplifter'], rotation=45)
    plt.yticks(tick_marks, ['non_shoplifter','shoplifter'])
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'))
    plt.close()

    print(f"✅ Confusion matrix saved to {os.path.join(PLOTS_DIR, 'confusion_matrix.png')}")
