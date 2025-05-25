# src/preprocessing.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import glob

IMAGE_SIZE = 224

def load_data(train_csv):
    df = pd.read_csv(train_csv)
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['soil_type'])
    class_names = label_encoder.classes_
    return df, label_encoder, class_names

def split_data(df, test_size=0.2):
    return train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

class SoilDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def find_image_path(self, base_name):
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            path = os.path.join(self.img_dir, base_name + ext)
            if os.path.exists(path):
                return path
        files = glob.glob(os.path.join(self.img_dir, base_name + ".*"))
        return files[0] if files else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        base_name = os.path.splitext(str(self.df.loc[idx, 'image_id']))[0]
        path = self.find_image_path(base_name)
        image = Image.open(path).convert("RGB")
        label = self.df.loc[idx, 'label']
        if self.transform:
            image = self.transform(image)
        return image, label
