eimport torch
from torchvision import datasets, models, transforms
from torch import nn, optim

class PlantDiseaseDetector(nn.Module):
    """
    Class for our dog detector model using pretrained resnet50
    """
    def __init__(self, num_classes):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        # for freezing the resnet model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Sequential(
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, num_classes)
        )


    def forward(self,x):
        return self.model(x)