import torch
import torch.nn as nn
import pytorch_lightning as pl
from data_utils import load_eb_dataset, get_raw_dataset_splits
from pytorch_lightning import LightningDataModule, Trainer
import torch.nn.functional as F
#Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self, num_classes, grayscale=False):
        super(LeNet5, self).__init__()
        
        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            
            nn.Conv2d(in_channels, 6*in_channels, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6*in_channels, 16*in_channels, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5*in_channels, 120*in_channels),
            nn.Tanh(),
            nn.Linear(120*in_channels, 84*in_channels),
            nn.Tanh(),
            nn.Linear(84*in_channels, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas
    
    
def get_model(model_name=None):
    return LeNet5(num_classes=10, grayscale=False)

def finetune_model(cfg):
    model = get_model()
    train_dataset, val_dataset, test_dataset = get_raw_dataset_splits(cfg)
    trainer = Trainer(devices=1, max_epochs=cfg.max_epochs)