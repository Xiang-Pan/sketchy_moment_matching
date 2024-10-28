import clip
import torch
import torch.nn as nn
from torch import set_grad_enabled

from .nets_utils import EmbeddingRecorder

class LinearCLIP(nn.Module):
    def __init__(self, channel, num_classes, im_size, record_embedding: bool = False, no_grad: bool = False,
                 pretrained: bool = False):
        super(LinearCLIP, self).__init__()
        self.model, _ = clip.load('ViT-B/32', device='cpu')
        self.classifier = nn.Linear(512, num_classes)
        self.freeze()
        self.train()

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    def get_last_layer(self):
        return self.classifier

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            x = self.featurizer(x)
            x = self.embedding_recorder(x)
            x = self.classifier(x)
        return x

    def featurizer(self, input):
        input_features = self.model.encode_image(input)
        input_features = input_features / input_features.norm(dim=-1, keepdim=True)
        return input_features

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
