import clip
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module


class LinearCLIP(Module):
    def __init__(self, model, num_classes=10):
        super(LinearCLIP, self).__init__()
        self.model, _ = clip.load('ViT-B/32', device='cpu')
        self.classifier = nn.Linear(512, num_classes)
        # nn.init.xavier_uniform_(self.classifier.weight)
        # nn.init.zeros_(self.classifier.bias)
        self.freeze()
        self.train()


    def featurizer(self, input):
        input_features = self.model.encode_image(input)
        input_features = input_features / input_features.norm(dim=-1, keepdim=True)
        return input_features

    def forward(self, input, out_feature=False):
        input_features = self.featurizer(input)
        logits = self.classifier(input_features)
        if out_feature:
            return input_features, logits
        else:
            return logits

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
            
    def train(self, mode=True):
        self.training = mode
        for module in self.model.children():
            module.train(False)     
        for module in self.classifier.children():
            module.train(mode)
        return self