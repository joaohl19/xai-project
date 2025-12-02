import torch
from torch import nn
import torch.nn.functional as F

# Defining a simple CNN architecture to classifying images between cat and dog
class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0, batch_normalization=True, classes=2):
        super().__init__()

        def maybe_batch_normalization(features):
            return nn.BatchNorm2d(features) if batch_normalization else nn.Identity()
        
        self.maxpool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout2d(dropout_rate)

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = maybe_batch_normalization(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = maybe_batch_normalization(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = maybe_batch_normalization(128)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = maybe_batch_normalization(256)

        self.classifier = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, classes)
        )
    def forward(self, input):
        input = self.maxpool(F.relu(self.bn1(self.conv1(input))))
        input = self.maxpool(F.relu(self.bn2(self.conv2(input))))
        input = self.maxpool(F.relu(self.bn3(self.dropout(self.conv3(input)))))
        input = self.maxpool(F.relu(self.bn4(self.conv4(input))))

        output = self.classifier(torch.flatten(input, 1))

        return output