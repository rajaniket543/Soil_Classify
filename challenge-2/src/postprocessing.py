import torch
import pandas as pd
from torch.utils.data import DataLoader
from src.model import get_model
from src.preprocessing import SoilDataset, get_transforms

def inference(model_path, test_dir, test_ids_csv, submission_path='submission.csv'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_transform = get_transforms()
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    test_dataset = SoilDataset(test_dir, test_ids_csv, transform=test_transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    preds = []
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            prob = torch.softmax(out, dim=1)[:, 1]
            preds.extend((prob >= 0.5).long().cpu().numpy())
    ids_df = pd.read_csv(test_ids_csv)
    ids_df['label'] = preds
    ids_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
