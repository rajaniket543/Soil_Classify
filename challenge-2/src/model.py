import torch.nn as nn
import torchvision.models as models

def get_model():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model
