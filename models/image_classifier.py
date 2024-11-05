import torch.nn as nn
from abc import ABC, abstractmethod

class ImageClassifier(nn.Module, ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, inputs):
        pass
    
    @abstractmethod
    def get_feature(self, inputs):
        pass