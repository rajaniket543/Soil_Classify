# src/postprocessing.py

import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

class TestDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.exts = ['.jpg', '.jpeg', '.png', '.webp', '.gif']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, 'image_id']
        base = os.path.splitext(img_id)[0]
        for ext in self.exts:
            path = os.path.join(self.img_dir, base + ext)
            if os.path.exists(path):
                image = Image.open(path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                return image, img_id
        return None, img_id

def generate_submission(model, test_loader, label_encoder, device, output_path="submission.csv"):
    model.eval()
    preds = []
    with torch.no_grad():
        for images, image_ids in test_loader:
            batch = [(img, id_) for img, id_ in zip(images, image_ids) if img is not None]
            if not batch:
                continue
            imgs_tensor, ids = zip(*batch)
            imgs_tensor = torch.stack(imgs_tensor).to(device)
            outputs = model(imgs_tensor)
            pred_labels = torch.argmax(outputs, dim=1).cpu().numpy()
            classes = label_encoder.inverse_transform(pred_labels)
            preds.extend(zip(ids, classes))

    pred_df = pd.DataFrame(preds, columns=["image_id", "soil_type"])
    return pred_df
