
import torch.nn as nn


class DLM_CBR_tiny(nn.Module):

    def __init__(self):
        super().__init__()

        self.dropout = 0.0
        self.feature_size = 512

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(256, 512, kernel_size=5, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.feature_size, 1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = self.flatten(x)

        return self.head(x)



