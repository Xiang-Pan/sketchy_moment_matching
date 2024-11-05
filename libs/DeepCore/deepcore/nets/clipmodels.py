import clip
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
from torch import set_grad_enabled
from .nets_utils import EmbeddingRecorder
from transformers import CLIPModel

# class LinearCLIP(nn.Module):
#     def __init__(self, channel, num_classes, im_size, record_embedding: bool = False, no_grad: bool = False,
#                  pretrained: bool = False):
#         super(LinearCLIP, self).__init__()
#         self.model, _ = clip.load('ViT-B/32', device='cpu')
#         self.classifier = nn.Linear(512, num_classes)
#         self.freeze()
#         self.train()

#         self.embedding_recorder = EmbeddingRecorder(record_embedding)
#         self.no_grad = no_grad

#     def get_last_layer(self):
#         return self.classifier

#     def forward(self, x):
#         with set_grad_enabled(not self.no_grad):
#             x = self.featurizer(x)
#             x = self.embedding_recorder(x)
#             x = self.classifier(x)
#         return x

#     def featurizer(self, input):
#         input_features = self.model.encode_image(input)
#         input_features = input_features / input_features.norm(dim=-1, keepdim=True)
#         return input_features

#     def freeze(self):
#         for param in self.model.parameters():
#             param.requires_grad = False
#         for param in self.classifier.parameters():
#             param.requires_grad = True

#     def train(self, mode=True):
#         self.training = mode
#         for module in self.model.children():
#             module.train(False)     
#         for module in self.classifier.children():
#             module.train(mode)
#         return self


class CLIP(Module):
    def __init__(self, num_classes: int,
                 record_embedding: bool = False,
                 no_grad: bool = False,
                 layers: str = "-1"
                 ):
        super(CLIP, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.model = self.model.float()
        self.classifier = nn.Linear(512, num_classes)
        print(self.model)
        if layers == "-1":
            self.freeze = self.freeze_N1
            self.train = self.train_N1
        elif layers == "-2":
            self.freeze = self.freeze_N2
            self.train = self.train_N2
        elif layers == "-3":
            self.freeze = self.freeze_N3
            self.train = self.train_N3
        else:
            raise ValueError("Invalid layers")
        self.freeze()
        self.train()
        self.model.encode_image = self.model.get_image_features
        
        print("!" * 50)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
        for name, param in self.classifier.named_parameters():
            if param.requires_grad:
                print(name)
        print("!" * 50)

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    def featurizer(self, input):
        outputs = self.model.encode_image(input)
        input_features = self.model.encode_image(input)
        input_features = input_features / input_features.norm(dim=-1, keepdim=True)
        return input_features

    def get_last_layer(self):
        return self.classifier

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            feature = self.featurizer(x)
            feature = self.embedding_recorder(feature)
            logit = self.classifier(feature)
        return logit

    def freeze_N1(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def train_N1(self, mode=True):
        self.training = mode
        for module in self.model.children():
            module.train(False)
        for module in self.classifier.children():
            module.train(mode)
        return self
    
    def freeze_N2(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = "visual_projection" in name
        for param in self.classifier.parameters():
            param.requires_grad = True

    def train_N2(self, mode=True):
        self.training = mode
        for module in self.model.children():
            module.train(False)
        #! NEED TO DOUBEL CHECK THIS
        for name, param in self.model.named_parameters():
            if "visual_projection" in name:
                param.requires_grad = mode
        for module in self.classifier.children():
            module.train(mode)
        return self

    def freeze_N3(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = ("visual_projection" in name) or ("vision_model.encoder.layers.11" in name)
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def train_N3(self, mode=True):
        self.training = mode
        for module in self.model.children():
            module.train(False)
        #! NEED TO DOUBEL CHECK THIS
        for name, param in self.model.named_parameters():
            if ("visual_projection" in name) or ("vision_model.encoder.layers.11" in name):
                param.requires_grad = mode
        for module in self.classifier.children():
            module.train(mode)
        return self
    


def LinearCLIP(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False, pretrained: bool = False):
    assert channel == 3, "channel must be 3"
    # assert pretrained == True, "pretrained must be True"
    if not pretrained:
        print("Warning!!!!!!!!!!: LinearCLIP is always pretrained.")
    assert im_size == (224, 224), "im_size must be (224, 224)"
    return CLIP(num_classes, record_embedding, no_grad, layers="-1")

def TwoLayerCLIP(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False, pretrained: bool = False):
    assert channel == 3, "channel must be 3"
    # assert pretrained == True, "pretrained must be True"
    if not pretrained:
        print("Warning!!!!!!!!!!: TwoLayerCLIP is always pretrained.")
    assert im_size == (224, 224), "im_size must be (224, 224)"
    return CLIP(num_classes, record_embedding, no_grad, layers="-2")

def ThreeLayerCLIP(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False, pretrained: bool = False):
    assert channel == 3, "channel must be 3"
    # assert pretrained == True, "pretrained must be True"
    if not pretrained:
        print("Warning!!!!!!!!!!: ThreeLayerCLIP is always pretrained.")
    assert im_size == (224, 224), "im_size must be (224, 224)"
    return CLIP(num_classes, record_embedding, no_grad, layers="-3")