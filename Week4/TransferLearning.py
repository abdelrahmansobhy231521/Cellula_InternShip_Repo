import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, jaccard_score
import pandas as pd
import segmentation_models_pytorch as smp

# ==== Dataset ====

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])
        self.labels = sorted([f for f in os.listdir(label_dir) if f.endswith('.tif') or f.endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.label_dir, self.labels[idx])

        try:
            image = tifffile.imread(img_path)
        except Exception as e:
            raise RuntimeError(f"Error reading image file {img_path}: {e}")

        if image.ndim == 2:
            image = image[np.newaxis, ...]  # (1, H, W)
        else:
            image = np.transpose(image, (2, 0, 1))  # (C, H, W)

        image = image.astype(np.float32)
        image = (image - image.mean()) / (image.std() + 1e-8)

        try:
            mask = tifffile.imread(mask_path)
        except Exception as e:
            raise RuntimeError(f"Error reading mask file {mask_path}: {e}")

        if mask.ndim == 3:
            mask = mask[:, :, 0]  # if mask is RGB, use one channel

        mask = (mask > 0).astype(np.float32)  # binarize
        mask = np.expand_dims(mask, axis=0)  # (1, H, W)

        return torch.tensor(image), torch.tensor(mask)

# ==== Metrics ====

def compute_metrics(preds, targets):
    preds_bin = (preds > 0.5).float().cpu().numpy().ravel()
    targets_bin = targets.cpu().numpy().ravel()
    f1 = f1_score(targets_bin, preds_bin, zero_division=1)
    acc = accuracy_score(targets_bin, preds_bin)
    iou = jaccard_score(targets_bin, preds_bin, zero_division=1)
    return f1, acc, iou

# ==== Training ====

def train_fn(loader, model, loss_fn, optimizer, device):
    model.train()
    total_loss, all_preds, all_targets = 0, [], []

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)

        preds = model(data)
        loss = loss_fn(preds, targets)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_preds.append(preds.detach())
        all_targets.append(targets.detach())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    f1, acc, iou = compute_metrics(all_preds, all_targets)

    return total_loss / len(loader), f1, acc, iou

# ==== Main ====

def main():
    # === Config ===
    base_path = r'D:\Cellula_tech_intern\Week4\data'
    image_dir = os.path.join(base_path, 'images')
    label_dir = os.path.join(base_path, 'labels')
    lr = 1e-3
    batch_size = 2
    epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Dataset and Loader ===
    dataset = SegmentationDataset(image_dir, label_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # === Determine input channels ===
    sample_img = tifffile.imread(os.path.join(image_dir, os.listdir(image_dir)[0]))
    in_channels = sample_img.shape[2] if sample_img.ndim == 3 else 1

    # === Load pretrained U-Net model ===
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=1,
        activation="sigmoid"
    ).to(device)

    # === Loss and Optimizer ===
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # === Training Loop ===
    logs = []
    for epoch in range(epochs):
        loss, f1, acc, iou = train_fn(loader, model, loss_fn, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | IoU: {iou:.4f}")
        logs.append({'epoch': epoch+1, 'loss': loss, 'f1_score': f1, 'accuracy': acc, 'iou': iou})

    # === Save Logs and Model ===
    pd.DataFrame(logs).to_csv("training_log.csv", index=False)
    torch.save(model.state_dict(), "unet_resnet34_pretrained.pth")
    print("Training complete. Model and logs saved.")

if __name__ == "__main__":
    main()
