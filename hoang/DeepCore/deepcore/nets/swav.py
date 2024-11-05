import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torch import set_grad_enabled
from .nets_utils import EmbeddingRecorder

class SwAVResNet50(nn.Module):
    def __init__(self, channel, num_classes, im_size, record_embedding: bool = True, no_grad: bool = False,
                 pretrained: bool = False):
        super(SwAVResNet50, self).__init__()
        backbone = torch.hub.load('facebookresearch/swav', 'resnet50')
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        self.linear = nn.Linear(2048, num_classes)
        self.freeze()
        self.train()
        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad
        
    def get_last_layer(self):
        return self.linear
            
    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            out = self.features(x)
            feature = out.view(out.size(0), -1)
            feature = self.embedding_recorder(feature)
            output = self.linear(feature)
        return output
    
    def freeze(self):
        for param in self.features.parameters():
            param.requires_grad = False

        for param in self.linear.parameters():
            param.requires_grad = True            
    def train(self, mode=True):
        self.training = mode
        for module in self.features.children():
            module.train(False)     
        for module in self.linear.children():
            module.train(mode)
        return self            
def test():
    net = SwAVResNet50(100)
    print(net)
    y = net(torch.randn(32, 3, 32, 32))
    print(y.size())

