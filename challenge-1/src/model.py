# src/model.py

import torch.nn as nn # type: ignore
import torchvision.models as models # type: ignore

def get_resnet_model(num_classes=4):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
