import timm
import torch
import torch.nn as nn
from torch import set_grad_enabled
from .nets_utils import EmbeddingRecorder

class TinyNetImageClassifier(nn.Module):
    def __init__(self,
                backbone_version='tinynet_e.in1k',
                num_classes=10,
                record_embedding: bool = False,
                no_grad: bool = False):
        super(TinyNetImageClassifier, self).__init__()
        self.model = timm.create_model(backbone_version, pretrained=True, num_classes=num_classes)
        
        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    # def forward(self, inputs):
    #     features = self.get_feature(inputs)
    #     logits = self.model.classifier(features)
    #     return logits

    def forward(self, inputs):
        with set_grad_enabled(not self.no_grad):
            feature = self.get_feature(inputs)
            feature = self.embedding_recorder(feature)
            logit = self.model.classifier(feature)
        return logit
    
    def get_feature(self, inputs):
        unpooled_features = self.model.forward_features(inputs)
        features = self.model.forward_head(unpooled_features, pre_logits=True)
        return features

    def get_transforms(self, is_training=False):
        data_config = timm.data.resolve_model_data_config(self.model)
        transforms = timm.data.create_transform(**data_config, is_training=is_training)
        return transforms


def TinyNet(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False, pretrained: bool = False):
    assert channel == 3, "channel must be 3"
    # assert pretrained == True, "pretrained must be True"
    if not pretrained:
        print("Warning!!!!!!!!!!: TinyNet is always pretrained.")
    assert im_size == (224, 224), "im_size must be (224, 224)"
    return TinyNetImageClassifier(num_classes=num_classes, record_embedding=record_embedding, no_grad=no_grad)
    

if __name__ == "__main__":
    model = TinyNetImageClassifier().cuda()
    from torch.utils.data import DataLoader, TensorDataset
    print(model)
    model.eval()
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    print(transforms)
    data = torch.randn(1, 3, 224, 224).cuda()
    output = model(data)
    # get total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    # get linear layer parameters
    total_linear_params = sum(p.numel() for p in model.model.classifier.parameters())
    print(f'{total_linear_params:,} total linear parameters.')
    print(output.shape)
    # print the model