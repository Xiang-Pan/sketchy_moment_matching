import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNForSVHNInner(nn.Module):
    def __init__(self):
        super(CNNForSVHNInner, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x, out_feature=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        if out_feature:
            return x, out
        return out

class CNNForSVHN(nn.Module):
    def __init__(self):
        super(CNNForSVHN, self).__init__()
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # Max Pooling 1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Convolutional Layer 4
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # Convolutional Layer 5
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Convolutional Layer 6
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # Max Pooling 2
        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)  # The input size needs adjustment based on the output of the last pooling layer
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)  # Assuming 10 classes for SVHN

    def forward(self, x, out_feature=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = x.view(-1, 128 * 4 * 4)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        # breakpoint()
        
        if out_feature:
            return x, out
        return out

class CNNForCIFAR10(nn.Module):
    def __init__(self):
        super(CNNForCIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x, out_feature=False):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        if out_feature:
            return x, out
        return out

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            #1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),   # 28*28->32*32-->28*28
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 14*14
            
            #2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 10*10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 5*5      
            
            nn.Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            # nn.Tanh(),                
        )
        self.classifier = nn.Linear(in_features=84, out_features=10)
     
    def forward(self, x, out_feature=False):
        x = self.feature(x)
        out = self.classifier(x)
        if out_feature:
            return x, out
        return out
