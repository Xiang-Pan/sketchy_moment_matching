import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from timm import create_model

class SwAVResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(SwAVResNet50, self).__init__()
        backbone = torch.hub.load('facebookresearch/swav', 'resnet50')
        # backbone = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
             
        
        self.linear = nn.Linear(2048, num_classes)
        self.freeze()
        self.train()
        
    def forward(self, x, out_feature=False):
        out = self.features(x)
        feature = out.view(out.size(0), -1)
        output = self.linear(feature)
        if out_feature:
            return feature, output
        else:
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

