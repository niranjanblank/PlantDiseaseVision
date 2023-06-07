import torch
from torchvision import datasets, models, transforms
from torch import nn, optim

class PlantDiseaseDetector(nn.Module):
    """
    Class for our dog detector model using pretrained resnet50
    """
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        # for freezing the resnet model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self,x):
        return self.model(x)