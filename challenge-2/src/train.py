import os
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from sklearn.metrics import f1_score
from tqdm import tqdm
from torchvision import transforms

from src.model import get_model
from src.preprocessing import SoilDataset, get_transforms

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, lbls in tqdm(loader, desc='Train'):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            output = model(imgs)
            total_loss += criterion(output, lbls).item() * imgs.size(0)
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
    f1 = f1_score(all_labels, all_preds)
    return total_loss / len(loader.dataset), f1

def train_model(df, image_dir, model_path='best_model.pth', epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform, val_transform = get_transforms()
    full_dataset = SoilDataset(image_dir, df, transform=train_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_f1 = 0.0
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_f1)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_path)
            print(f"→ Saved best model with F1: {best_f1:.4f}")
