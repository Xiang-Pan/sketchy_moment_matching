import clip
import torch.nn as nn
from models.litclassifier import ResNet
from models.clip_models import CLIPImageClassifier
# ImageEncoder


def get_net(cfg, num_classes):
    if "clip" in cfg.backbone.name:
        net = CLIPImageClassifier(backbone_version=cfg.backbone.version, num_classes=num_classes)
    elif "tinynet" in cfg.backbone.name:
        from models.tinynet_models import TinyNetImageClassifier
        net = TinyNetImageClassifier(backbone_version=cfg.backbone.version, num_classes=num_classes)
    elif "resnet" in cfg.backbone.name:
        resnet_params = {
            "backbone_name": cfg.backbone.name,
            "version": cfg.backbone.version,
            "num_classes": num_classes,
            "classifier_name": cfg.backbone.classifier
        }
        if "swav" in cfg.backbone.version:
            resnet_params["skip_pool"] = cfg.backbone.skip_pool
            resnet_params["pretrain_path"] = cfg.backbone.pretrain_path
        net = ResNet(**resnet_params)
    else:
        raise NotImplementedError

    return net