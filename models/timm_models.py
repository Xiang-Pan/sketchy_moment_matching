# timm/tinynet_e.in1k
import timm
import torch
import torch.nn as nn
import sys
sys.path.append('.')
# from models.image_classifier import ImageClassifier

class TimmImageClassifier(nn.Module):
    def __init__(self,
                 backbone_version='tinynet_e.in1k',
                 num_classes=10):
        super(TimmImageClassifier, self).__init__()
        self.model = timm.create_model(backbone_version, pretrained=True, num_classes=num_classes)

    def forward(self, inputs):
        features = self.get_feature(inputs)
        logits = self.model.classifier(features)
        return logits
    
    def get_feature(self, inputs):
        unpooled_features = self.model.forward_features(inputs)
        features = self.model.forward_head(unpooled_features, pre_logits=True)
        return features

    def get_transofrms(self, is_training=False):
        data_config = timm.data.resolve_model_data_config(self.model)
        transforms = timm.data.create_transform(**data_config, is_training=is_training)
        return transforms

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