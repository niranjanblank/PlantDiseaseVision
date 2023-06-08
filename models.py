import torch
from torchvision import datasets, models, transforms
from torch import nn, optim

class PlantDiseaseDetector(nn.Module):
    """
    Class for our dog detector model using pretrained resnet50
    """
    def __init__(self, num_classes):
        super().__init__()
        weights = models.ResNet50_Weights
        self.model = models.resnet50(weights=weights)
        # for freezing the resnet model
        for param in self.model.parameters():
            param.requires_grad = False
        self.fc1 = nn.Linear(2048, 1024)
        self.relu = nn.ReLU()
        self.dout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, num_classes)


    def forward(self,x):
        x = self.model(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dout(x)
        x = self.bn1(x)
        return self.fc2(x)