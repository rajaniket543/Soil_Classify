import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

def create_negatives(neg_dir, neg_labels_path, num_neg=1000):
    """
    Save CIFAR-10 images as negative samples.
    """
    os.makedirs(neg_dir, exist_ok=True)
    cifar = CIFAR10(root='data', train=True, download=True)
    rows = []
    for idx in tqdm(range(num_neg), desc='Creating negatives'):
        img, _ = cifar[idx]
        img_id = f"neg_{idx:04d}"
        img.save(os.path.join(neg_dir, f"{img_id}.jpg"))
        rows.append({'image_id': img_id, 'label': 0})
    neg_df = pd.DataFrame(rows)
    neg_df.to_csv(neg_labels_path, index=False)

def load_and_merge_labels(labels_path, neg_labels_path, output_path):
    """
    Merge soil and negative labels into a single CSV.
    """
    soil_df = pd.read_csv(labels_path).rename(columns={'soil_type':'label'})
    soil_df['label'] = 1
    neg_df = pd.read_csv(neg_labels_path)
    full_df = pd.concat([soil_df[['image_id','label']], neg_df], ignore_index=True)
    full_df.to_csv(output_path, index=False)
    return full_df

def merge_image_folders(soil_dir, neg_dir, output_dir):
    """
    Combine soil and negative images into one directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(soil_dir):
        src = os.path.join(soil_dir, fname)
        dst = os.path.join(output_dir, fname)
        if not os.path.exists(dst):
            os.link(src, dst)

    for fname in os.listdir(neg_dir):
        src = os.path.join(neg_dir, fname)
        dst = os.path.join(output_dir, fname)
        if not os.path.exists(dst):
            os.link(src, dst)

class SoilDataset(Dataset):
    def __init__(self, image_dir, labels_df, transform=None, is_test=False):
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test

        if not is_test:
            self.df = labels_df.copy()
        else:
            self.df = pd.read_csv(labels_df)
            self.df.columns = ['image_id']
            self.df['label'] = -1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import torch
        row = self.df.iloc[idx]
        img_id = str(row.image_id)

        image = None
        for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            fname = img_id if img_id.lower().endswith(ext) else img_id + ext
            path = os.path.join(self.image_dir, fname)
            if os.path.exists(path):
                image = Image.open(path).convert('RGB')
                break

        if image is None:
            raise FileNotFoundError(f"Image {img_id} not found with any supported extension in {self.image_dir}")

        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row.label, dtype=torch.long)
        return image, label

def get_transforms():
    """
    Return training and validation transforms.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    return train_transform, val_transform
