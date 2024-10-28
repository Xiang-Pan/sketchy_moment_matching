import lightning as pl
import torchmetrics
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import LightningModule, LightningDataModule
from models.clip_models import CLIPImageClassifier
from pytorch_lightning import LightningModule
from torch.optim import SGD
import clip
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class EmbeddingRecorder(nn.Module):
    def __init__(self, record_embedding: bool = False):
        super().__init__()
        self.record_embedding = record_embedding

    def forward(self, x):
        if self.record_embedding:
            self.embedding = x
        return x

    def __enter__(self):
        self.record_embedding = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.record_embedding = False

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, activation="relu"):
        super().__init__()
        self.fc0 = nn.Linear(in_dim, hidden_dim)
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "tanh":
            self.act = nn.Tanh()
        else:
            self.act = nn.Identity()
        self.fc1 = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        x = self.fc0(x)
        x = self.act(x)
        x = self.fc1(x)
        return x


#! never create reference
class ResNet(nn.Module):
    def __init__(self,
                    backbone_name="resnet50",
                    version="IMAGENET1K_V2",
                    num_classes=10,
                    classifier_name="mlp_50",
                    skip_pool=None,
                    pretrain_path=None,
                    deepcore=False
                 ):
        super().__init__()
        logger.info(f"backbone_name: {backbone_name}")
        logger.info(f"version: {version}")
        logger.info(f"num_classes: {num_classes}")
        if backbone_name == "resnet50":
            if "swav" in version:
                from models.swav_models import get_model
                print(pretrain_path)
                net = get_model("resnet50", skip_pool=skip_pool, pretrain_path=pretrain_path)
                net = net[0]
                feature_dim = 2048
            else:
                net = torchvision.models.resnet50(weights=version)
                feature_dim = 2048
        elif backbone_name == "resnet18":
            net = torchvision.models.resnet18(weights=version)
            feature_dim = 512
        else:
            raise NotImplementedError
        self.net = net
        self.classifier_name = classifier_name
        if classifier_name == "mlp_50":
            self.net.fc = MLP(feature_dim, 50, num_classes).cuda()
        elif classifier_name == "linear":
            self.net.fc = nn.Linear(feature_dim, num_classes).cuda()
        else:
            raise NotImplementedError
        if deepcore:
            self.embedding_recorder = EmbeddingRecorder(False)
        else:
            self.embedding_recorder = None
    
    def get_last_layer(self):
        return self.net.fc

    def get_feature_encoder(self):
        return nn.Sequential(*list(self.net.children())[:-1])

    def forward(self, x):
        feature = self.get_feature_encoder()(x)
        feature = feature.view(feature.size(0), -1)
        if self.embedding_recorder:
            feature = self.embedding_recorder(feature)
        logit = self.get_last_layer()(feature)
        return logit

    def get_feature(self, x):
        batch_size = x.size(0)
        feat = self.get_feature_encoder()(x)
        feat = feat.view(batch_size, -1)
        if self.classifier_name == "mlp_50":
            feat = self.net.fc.fc0(feat)
            feat = self.net.act(feat)
        return feat


class LitClassifier(LightningModule):
    def __init__(self, cfg, max_epochs=50):
        super().__init__()
        self.cfg = cfg
        self.max_epochs = max_epochs
        num_classes = cfg.dataset.num_classes
        self.lr = cfg.finetuning.optimizer.lr
        logger.info(f"Configuration: {cfg.backbone}")
        self.net = self.get_net(cfg, num_classes)
        # logger.info(self.net)
        self.get_feature = lambda x: self.net.encode_image(x) / self.net.encode_image(x).norm(dim=-1, keepdim=True) if "clip" in cfg.backbone.name else self.net.get_feature(x)
        self.acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.f1_metric = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes, average="macro")

        #! TODO: make sure this work for retraining
        backbone_name = cfg.backbone.name
        if backbone_name == "resnet18":
            second_layer_names = ["layer4.1.conv2"]
            fc_name = ["fc"]
        elif backbone_name == "resnet50":
            second_layer_names = ["layer4.2.conv3"]
            fc_name = ["fc"]
        elif backbone_name == "clip-vit-base-patch32":
            #* make sure
            third_layer_names = ["image_encoder.vision_model.encoder.layers.11"]
            second_layer_names = ["image_encoder.vision_model.visual_projection"]
            fc_name = ["classification_head"]
        elif backbone_name == "tinynet_e":
            second_layer_names = ["conv_head"]
            fc_name = ["classifier"]
        else:
            logger.info(backbone_name)
            raise NotImplementedError
        logger.debug(self.net)
        if hasattr(cfg, "finetuning") and hasattr(cfg.finetuning, "layers"):
            logger.info(f"cfg.finetuning.layers: {cfg.finetuning.layers}")
            if cfg.finetuning.layers == "all":
                for name, param in self.net.named_parameters():
                    param.requires_grad = True
                logger.info("All layers are finetuned")
            else:
                if cfg.finetuning.layers == -3:
                    layer_names = third_layer_names + second_layer_names + fc_name
                if cfg.finetuning.layers == -2:
                    layer_names = second_layer_names + fc_name
                elif cfg.finetuning.layers == -1:
                    layer_names = fc_name
                finetuning_layers = []
                for name, param in self.net.named_parameters():
                    if any(layer_name in name for layer_name in layer_names):
                        finetuning_layers.append(name)
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                logger.info(f"layer_names: {layer_names}")
                logger.info(f"finetuning_layers: {finetuning_layers}")
        self.best_val_acc = -1
    
    def get_net(self, cfg, num_classes):
        if "clip" in cfg.backbone.name:
            from models.clip_models import CLIPImageClassifier
            net = CLIPImageClassifier(backbone_version=cfg.backbone.version, num_classes=num_classes, process_images=True)
        elif "tinynet_e" in cfg.backbone.name:
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

    def get_feature_layers(self):
        # get the layers before requires_grad
        layers = []
        for name, module in self.net.named_modules():
            if all([not param.requires_grad for name, param in module.named_parameters()]):
                layers.append(module)
        return nn.Sequential(*layers)
    
    def finetuning_layers(self):
        layers = []
        for name, module in self.net.named_modules():
            if all([param.requires_grad for name, param in module.named_parameters()]):
                layers.append(module)
        return nn.Sequential(*layers)

    def get_feature(self, x):
        feature = self.get_feature_layers()(x)
        return feature

    def forward(self, x):
        logit = self.net(x)
        return logit

    def _step(self, batch, prefix):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = self.acc_metric(y_hat, y)
        f1 = self.f1_metric(y_hat, y)
        self.log(f'{prefix}/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{prefix}/acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{prefix}/lr', self.lr, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{prefix}/f1', f1, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, 'test')

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        val_acc = self.trainer.callback_metrics.get('val/acc')
        self.best_val_acc = max(val_acc, self.best_val_acc)
        logger.debug(f"best_val_acc: {self.best_val_acc.item()}")
        self.log("val/best_acc",self.best_val_acc)

    def configure_optimizers(self):
        if self.cfg.finetuning.optimizer.type == "SGD":
            lr = self.lr
            momentum = self.cfg.finetuning.optimizer.momentum
            weight_decay = self.cfg.finetuning.optimizer.weight_decay
            optimizer = SGD(self.net.parameters(), lr=self.lr, momentum=momentum, weight_decay=weight_decay)
        elif self.cfg.finetuning.optimizer.type == "Adam":
            lr = self.lr
            weight_decay = self.cfg.finetuning.optimizer.weight_decay
            feature_lr_decay = self.cfg.finetuning.optimizer.feature_lr_decay
            cls_params = list(map(id, self.net.get_last_layer().parameters()))
            encoder_params = list(filter(lambda p: id(p) not in cls_params and p.requires_grad, self.net.parameters()))
            
            if  hasattr(self.cfg.finetuning.optimizer, "feature_weight_decay"):
                feature_weight_decay = self.cfg.finetuning.optimizer.feature_weight_decay
                optimizer = torch.optim.Adam([
                    {"params": self.net.get_last_layer().parameters(), "lr": lr, "weight_decay": weight_decay},
                    {"params": encoder_params, "lr": lr * feature_lr_decay, "weight_decay": feature_weight_decay},
                ])
            else:
                optimizer = torch.optim.Adam([
                    {"params": self.net.get_last_layer().parameters(), "lr": lr, "weight_decay": weight_decay},
                    {"params": encoder_params, "lr": lr * feature_lr_decay, "weight_decay": weight_decay},
                ])
            logger.info(f"feature_lr_decay: {feature_lr_decay}")
            logger.info(optimizer)
        else:
            raise NotImplementedError
        if hasattr(self.cfg.finetuning, "scheduler"):
            if self.cfg.finetuning.scheduler.type == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
            elif self.cfg.finetuning.scheduler.type == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg.finetuning.scheduler.step_size, gamma=self.cfg.finetuning.scheduler.gamma)
            else:
                raise NotImplementedError
        else:
            scheduler = None
        logger.info(f"Optimizer: {optimizer}")
        logger.info(f"Scheduler: {scheduler}")
        if scheduler:
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        else:
            return [optimizer]