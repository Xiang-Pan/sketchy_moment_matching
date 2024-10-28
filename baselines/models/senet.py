import torch
import torch.nn as nn

class SeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SeNet, self).__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=False)
        self.linear1 = nn.Linear(1000, 256)
        self.linear2 = nn.Linear(256, num_classes)
    
    def forward(self, x, out_feature=False):
        x = self.backbone(x)
        # x = F.relu(self.linear1(x))
        x = self.linear1(x)
        output = self.linear2(x)
        if out_feature:
            return x, output
        else:
            return output


def test():
    net = SeNet()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
