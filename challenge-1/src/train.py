# src/train.py

import torch
from sklearn.metrics import f1_score
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=1e-4):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_f1 = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        f1 = f1_score(all_labels, all_preds, average=None)
        min_f1 = f1.min()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Min F1: {min_f1:.4f}")

        if min_f1 > best_f1:
            best_f1 = min_f1
            torch.save(model.state_dict(), "best_soil_model.pth")
            print("✅ Model saved.")
