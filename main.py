import hydra
import pickle as pkl
from omegaconf import OmegaConf
from sklearn.neural_network import MLPClassifier
import submitit
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
import torchvision
import torch
import yaml
import wandb
import os
import clip
from sklearn.model_selection import GridSearchCV
from PIL import Image
from tqdm.auto import tqdm
import torchmetrics
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import LightningModule, LightningDataModule
from torchvision.models import ResNet18_Weights, ResNet50_Weights
import logging
from lightning.pytorch.callbacks import BatchSizeFinder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import libs.DeepCore.deepcore.methods as methods

import numpy as np
logger = logging.getLogger(f"./logs/{__name__}.log")
logger.setLevel(logging.DEBUG)
# set level
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import PredefinedSplit
# import clearml
# from clearml import Task
from utils.data_utils import get_raw_dataset, get_raw_dataset_splits

def get_influence_score_batch(model_path, 
                            all_data_path,
                            batch_idx, 
                            batch_size=100,
                            save_path="influence_score"):
    model = torch.load(model_path)
    all_data = torch.load(all_data_path)
    features, labels = all_data
    model = model.cuda()
    features = features.cuda()
    labels = labels.cuda()
    all_data = (features.cuda(), labels.cuda())

    influence_score_list = []
    for idx in tqdm(range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(features)))):
        data, label = features[idx], labels[idx]
        from libs.code_datasetptuning.calc_influence_function import calc_s_test_single
        influence_score = calc_s_test_single(model, data, label, all_data, gpu=0, recursion_depth=500, r=1)
        influence_score_list.append(influence_score)
    torch.save(influence_score_list, f"{save_path}/influence_score-batch_idx={batch_idx}-batch_size={batch_size}.pt")
    return influence_score_list

def load(path, map_location="cpu"):
    if os.path.exists(path):
        return torch.load(path, map_location=map_location)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        REMOTE = os.getenv("REMOTE")
        REMOTE_LABROOT = os.getenv("REMOTE_LABROOT")
        os.system(f"rsync -avzrP {REMOTE}:{REMOTE_LABROOT}/data_pruning/{path} {os.path.dirname(path)}")
        return torch.load(path, map_location=map_location)


class EBDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.data = X
        self.targets = Y
        self.classes = torch.unique(Y)
        # sort the classes
        self.classes = self.classes.sort()[0]
        self.num_classes = len(self.classes)
        self.targets = Y
        self.transform = None
        self.target_transform = None
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y


def get_eb_dataset_str(folder:str, phase:str="train"):
    X = load(f"{folder}/{phase}/X.pt", map_location="cpu")
    Y = load(f"{folder}/{phase}/Y.pt", map_location="cpu")
    return EBDataset(X, Y)


def get_eb_dataset(cfg: OmegaConf, phase:str="train"):
    dataset_name = cfg.dataset.name
    original_eb_folder = f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}/phase={phase}"
    if cfg.pretraining is not None:
        original_eb_folder = f"{original_eb_folder}/pretraining#{cfg.pretraining.str}"
    if dataset_name == "cifar10":
        X = load(f"{original_eb_folder}/X.pt", map_location="cpu")
        Y = load(f"{original_eb_folder}/Y.pt", map_location="cpu")
        dataset = EBDataset(X, Y)
    elif dataset_name == "cifar100":
        X = load(f"{original_eb_folder}/X.pt", map_location="cpu")
        Y = load(f"{original_eb_folder}/Y.pt", map_location="cpu")
        dataset = EBDataset(X, Y)
    return dataset

class CustomSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, trasnform=None, target_transform=None):
        self.dataset = dataset
        self.indices = indices
        self.classes = dataset.classes
        self.num_classes = dataset.num_classes
        
        self.data = dataset.data
        self.targets = dataset.targets
        
        self.transform = trasnform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        x = self.data[self.indices[idx]]
        if self.transform is not None:
            x = self.transform(x)
        y = self.targets[self.indices[idx]]
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y

# def split_train_val(trainval_dataset, val_ratio):
#     # shuffle the indices
#     global_seed = torch.initial_seed()
#     torch.manual_seed(cfg.dataset.seed)
#     subset_indices = torch.arange(len(trainval_dataset))
#     subset_indices = torch.randperm(len(subset_indices))
#     train_ratio = 1 - val_ratio
#     train_indices = subset_indices[:int(len(subset_indices) * train_ratio)]
#     val_indices = subset_indices[int(len(subset_indices) * train_ratio):]
#     torch.manual_seed(global_seed)
#     return train_indices, val_indices

# def get_raw_dataset_splits(cfg):
#     trainval_dataset = get_raw_dataset(cfg, "train")
#     test_dataset = get_raw_dataset(cfg, "test")
#     val_dataset = get_raw_dataset(cfg, "val")
#     train_indices, val_indices = split_train_val(trainval_dataset, cfg.dataset.val_ratio)
#     train_dataset = torch.utils.data.Subset(trainval_dataset, train_indices)
#     val_dataset = torch.utils.data.Subset(trainval_dataset, val_indices)
#     return train_dataset, val_dataset, test_dataset


class DataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        try:
            self.train_batch_size = cfg.training.train_batch_size
            self.val_batch_size = cfg.training.val_batch_size
            self.test_batch_size = cfg.training.test_batch_size
        except:
            self.train_batch_size = 512
            self.val_batch_size = 512
            self.test_batch_size = 512
        self.debug = cfg.debug
        self.setup(stage=None)

    def setup(self, stage):
        cfg = self.cfg
        def get_eb_folder(cfg):
            eb_folder = f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}"
            if cfg.selection.rep is not None:
                eb_folder = f"{eb_folder}/rep#{cfg.selection.rep}"
            return eb_folder

        #* split into train and val

        log_dict = {}
        #!mode_1
        if self.cfg.mode in ["pretraining"]:
            train_dataset, val_dataset, test_dataset = get_raw_dataset_splits(self.cfg)
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset
        
        #!mode_2
        elif self.cfg.mode in ["feature_extraction"]:
            train_dataset, val_dataset, test_dataset = get_raw_dataset_splits(self.cfg)
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset

        #!mode_3
        elif self.cfg.mode in ["selection"]:
            if self.cfg.selection.data_type == "eb":
                try:
                    rep = self.cfg.training.rep
                except:
                    rep = None
                if rep is not None:
                    eb_folder = get_eb_folder(cfg)
                    self.train_dataset = get_eb_dataset_str(eb_folder, "train")
                    self.val_dataset = get_eb_dataset_str(eb_folder, "val")
                    self.test_dataset = get_eb_dataset_str(eb_folder, "test")
                else:
                    self.train_dataset = get_eb_dataset(self.cfg, "train")
                    self.val_dataset = get_eb_dataset(self.cfg, "val")
                    self.test_dataset = get_eb_dataset(self.cfg, "test")
                self.trainval_dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.val_dataset])
                
                
            elif self.cfg.selection.data_type == "raw":
                self.train_dataset = get_raw_dataset(self.cfg, "train")
                self.test_dataset = get_raw_dataset(self.cfg, "test")

        #! check mode_2 or mode_5
        elif self.cfg.mode in ["linear_probing"]:
            cfg = self.cfg
            try:
                rep = self.cfg.training.rep
            except:
                rep = None

            selection_folder = f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}/selection#{cfg.selection.str}"
            eb_folder = f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}"
            
            if rep is not None:
                eb_folder = f"{eb_folder}/rep#{rep}"
            cfg = self.cfg
            logger.debug("setup linear probing")
            if rep is not None:
                self.train_dataset = get_eb_dataset_str(eb_folder, "train")
                self.val_dataset = get_eb_dataset_str(eb_folder, "val")
                self.test_dataset = get_eb_dataset_str(eb_folder, "test")
            else:
                self.train_dataset = get_eb_dataset(cfg, "train")
                self.val_dataset = get_eb_dataset(cfg, "val")
                self.test_dataset = get_eb_dataset(cfg, "test")
            
            if self.cfg.selection.method == "full":
                selection_indices = torch.arange(len(self.train_dataset))
            else:
                selection_indices = load(f"{selection_folder}/fraction={cfg.selection.fraction}/train_index.pt")
            
            if len(selection_indices) > len(self.train_dataset) * self.cfg.selection.fraction:
                selection_indices = selection_indices[:int(len(self.train_dataset) * self.cfg.selection.fraction)]
            assert len(selection_indices) <= len(self.train_dataset) * self.cfg.selection.fraction, f"len(selection_indices): {len(selection_indices)}, len(self.train_dataset) * self.cfg.selection.fraction: {len(self.train_dataset) * self.cfg.selection.fraction}"
            # wandb.run.summary["subset_size"] = len(subset_indices)
            #! sample weight bug
            # if self.cfg.training.sample_weight == "s":
            #     s = torch.load(f"{folder}/s.pt")
            #     sample_weight = s[selection_indices.cpu()]
            #     self.sample_weight = sample_weight.detach().cpu()
            # else:
            #     self.sample_weight = None
            if type(selection_indices) == torch.Tensor:
                selection_indices = selection_indices.detach().cpu()
            elif type(selection_indices) == dict:
                selection_indices = selection_indices["indices"]
            logger.info(f"selection_indices.shape: {selection_indices.shape}")
            self.train_selection_indices = selection_indices
            self.train_dataset = CustomSubset(self.train_dataset, self.train_selection_indices)
            # make sure subset index <= (fraction * len(train_dataset))
            log_dict["subset_size"] = len(selection_indices)
            
        return log_dict
            
        

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )


class FeatureExtractionModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        from models.litclassifier import ResNet
        num_classes = 1 if cfg.dataset.name == "utk" else cfg.dataset.num_classes
        if "tinyclip" in cfg.backbone.name:
            from transformers import CLIPModel
            self.net = CLIPModel.from_pretrained("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")
            self.net.get_feature = self.net.get_image_features
        elif "clip" in cfg.backbone.name:
            from transformers import CLIPModel
            self.net = CLIPModel.from_pretrained(cfg.backbone.version)
            self.net.encode_image = self.net.get_image_features
            # self.net, _ = clip.load(cfg.backbone.version)
        elif "tinynet" in cfg.backbone.name:
            from models.timm_models import TimmImageClassifier
            self.net = TimmImageClassifier(backbone_version=cfg.backbone.version, num_classes=num_classes)
        elif "vit_tiny" in cfg.backbone.name:
            from models.timm_models import TimmImageClassifier
            self.net = TimmImageClassifier(backbone_version=cfg.backbone.version, num_classes=num_classes)
        elif "swav" in cfg.backbone.version:
                self.net = ResNet(backbone_name=cfg.backbone.name,
                            version=cfg.backbone.version,
                            num_classes=num_classes,
                            classifier_name=cfg.backbone.classifier,
                            skip_pool=cfg.backbone.skip_pool,
                            pretrain_path=cfg.backbone.pretrain_path)
        elif "resnet" in cfg.backbone.name:
            self.net = ResNet(backbone_name=cfg.backbone.name,
                        version=cfg.backbone.version,
                        num_classes=num_classes,
                        classifier_name=cfg.backbone.classifier)
        else:
            raise NotImplementedError
            
        self.backbone_name = cfg.backbone.name
        self.criterion = nn.CrossEntropyLoss() if cfg.dataset.name != "utk" else nn.MSELoss()
        self.dataset_phase = "train"

    def forward(self, x):
        return self.net.get_feature(x)

    def on_test_start(self):
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        if "tinyclip" in self.backbone_name:
            eb = self.net.get_image_features(x)
            eb = eb / eb.norm(dim=-1, keepdim=True)
            self.test_outputs.append((eb, y))
            return eb
        elif "clip" in self.backbone_name:
            eb = self.net.encode_image(x)
            eb = eb / eb.norm(dim=-1, keepdim=True)
            self.test_outputs.append((eb, y))
            return eb
        elif "tinynet" in self.backbone_name:
            eb = self.net.get_feature(x)
            self.test_outputs.append((eb, y))
            return eb
        elif "vit_tiny" in self.backbone_name:
            eb = self.net.get_feature(x)
            self.test_outputs.append((eb, y))
            return eb
        else:
            eb = self.net.get_feature(x)
            #TODO: make this clearner
            cls = self.net.net.fc
            if type(cls) == nn.Sequential:
                eb = cls[0](eb).detach().cpu()
            self.test_outputs.append((eb, y))
            return eb

    def on_test_end(self):
        cfg = self.cfg
        # if pretraining is not None:
        if hasattr(cfg, "pretraining"):
            path = f"cached_datasets/backbone#{self.cfg.backbone.str}/dataset#{self.cfg.dataset.str}/pretraining#{self.cfg.pretraining.str}/phase={self.dataset_phase}"
        else:
            path = f"cached_datasets/backbone#{self.cfg.backbone.str}/dataset#{self.cfg.dataset.str}/phase={self.dataset_phase}"
        os.makedirs(path, exist_ok=True)
        # calculate
        X = torch.cat([eb for eb, _ in self.test_outputs], dim=0)
        Y = torch.cat([y for _, y in self.test_outputs], dim=0)
        print(f"X.shape: {X.shape}")
        print(f"Y.shape: {Y.shape}" )
        torch.save(X, f"{path}/X.pt")
        torch.save(Y, f"{path}/Y.pt")

def orthogonal_projection(A):
    # A: d x d
    # we assume A is symmetric
    # set default device
    P = torch.zeros_like(A).to(A.device)
    P = torch.eye(A.shape[0]).to(A.device) - A @ torch.pinverse(A.T @ A + 1e-6 * torch.eye(A.shape[0]).to(A.device)) @ A.T
    return P


class OrthogonalRegularizer(nn.Module):
    def __init__(self, weight=1.0, reduction="mean"):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        # \min_S \min_W \| \Sigma_{\Phi(XS)}W - \Sigma_{\Phi(X)}\|^2_F
        
    
    def forward(self, X_train, X_val, S, current_epoch=0):
        # do the iterative optimization
        if current_epoch % 2 == 0:
            S.requires_grad_(True)
        else:
            S.requires_grad_(False)
        # XS is train
        # \min_S \min_W \| \Sigma_{\Phi(XS)}W - \Sigma_{\Phi(X)}\|^2_F
        # using theseus for bilevel optimization
        Sigma_train = X_train.T @ X_train / X_train.shape[0]
        train_selection = S @ X_train
        Sigma_train_selection = train_selection.T @ train_selection / train_selection.shape[0]
        Sigma_val = X_val.T @ X_val / X_val.shape[0]
        # low_level_problem = th.AutoDiffCostFunction(
        #     lambda W: torch.norm(Sigma_train_selection @ W - Sigma_val, p="fro") ** 2 / Sigma_val.shape[0],
        #     inputs=[S],
        #     outputs=["loss"],
        #     name="low_level_problem",
        # )
        
        # reg = cal_orthogonal_regularization(X_train, X_val)
        return 

def cal_orthogonal_regularization(X_train, X_val):
    n_train, p = X_train.shape
    n_val, _ = X_val.shape
    Sigma_train = X_train.T @ X_train / n_train
    Sigma_val = X_val.T @ X_val / n_val
    Sigma_train = Sigma_train.to("cuda")
    Sigma_val = Sigma_val.to("cuda")
    P_Sigma_train = orthogonal_projection(Sigma_train).to(Sigma_val.device)
    
    reg = torch.norm(P_Sigma_train @ Sigma_val, p="fro") 
    reg = reg / p
    return reg

class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features"""
    def __init__(self, num_labels, feat_dim, use_bn=True, reinit_head=True, orthogonal_regularization=False):
        super(RegLog, self).__init__()

        self.bn = None
        if use_bn:
            self.bn = nn.BatchNorm1d(feat_dim)

        self.linear = nn.Linear(feat_dim, num_labels)
        self.criterion = nn.CrossEntropyLoss()
        if reinit_head:
            print('reinit head weights by gaussian(0, 0.01). To be abandoned.')
            self.linear.weight.data.normal_(mean=0.0, std=0.01)
            self.linear.bias.data.zero_()
        if orthogonal_regularization:
            self.orthogonal_regularizer = OrthogonalRegularizer()
        else:
            self.orthogonal_regularizer = None

    def forward(self, x):
        # optional BN
        if self.bn is not None:
            x = self.bn(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)

    def get_loss(self, X_train, X_val, y_train):
        y_loss = self.criterion(self.forward(X_train), y_train)
        if self.orthogonal_regularizer is not None:
            return self.orthogonal_regularizer(X_train, X_val) + y_loss
        else:
            return y_loss

# def step_function(x):
    # return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

def differentiable_step_function(x, steepness=10):
    return torch.sigmoid(steepness * x)

class RegSelection(nn.Module):
    def __init__(self, num_labels, feat_dim, use_bn=True, reinit_head=True):
        super().__init__()
        self.bn = None 
        if use_bn:
            self.bn = nn.BatchNorm1d(feat_dim)
        self.linear = nn.Linear(feat_dim, num_labels)
        self.criterion = nn.CrossEntropyLoss()
        if reinit_head:
            print('reinit head weights by gaussian(0, 0.01). To be abandoned.')
            self.linear.weight.data.normal_(mean=0.0, std=0.01)
            self.linear.bias.data.zero_()
        D = 49500
        self.selection_vector = nn.Parameter(torch.randn(D)).requires_grad_(True)
        self.version = 2
        
    def forward(self, x):
        n_train, p = x.shape
        # optional BN
        if self.bn is not None:
            x = self.bn(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)
    
    
    def get_reg_loss(self, X_train, X_val):
        # diag
        if self.version == 1:
            sel_vec = differentiable_step_function(self.selection_vector)
            sel_mat = torch.diag(sel_vec).to(X_train.device)
            X_selection = sel_mat @ X_train
            orth_reg_loss = cal_orthogonal_regularization(X_selection, X_val)
            loss_dict = {
                "orth_reg_loss": orth_reg_loss,
            }
        elif self.version == 2:
            sel_mat = torch.diag(self.selection_vector).to(X_train.device)
            X_selection = sel_mat @ X_train
            orth_reg_loss = cal_orthogonal_regularization(X_selection, X_val)
            # selection_vector should be within [0, 1]
            bounded_reg_loss = (self.selection_vector - 1).clamp(min=0).sum() + (-self.selection_vector).clamp(min=0).sum()
            # selection_loss
            selection_loss = torch.norm(self.selection_vector, p=1)
            loss_dict = {
                "orth_reg_loss": orth_reg_loss,
                "bounded_reg_loss": bounded_reg_loss * 10,
                "selection_loss": selection_loss,
            }
            
        return loss_dict

    def fit(self, X_train, X_val, y_train):
        max_epochs = 100
        lr = 0.1
        optimizer = torch.optim.SGD([self.selection_vector], lr=lr)
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            sel_vec = differentiable_step_function(self.selection_vector)
            sel_mat = torch.diag(sel_vec).to(X_train.device)
            X_selection = sel_mat @ X_train
            # y_loss = self.criterion(self.forward(X_selection), y_train)
            reg_loss_dict = self.get_reg_loss(X_selection, X_val)
            loss = sum(reg_loss_dict.values())
            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch}, loss: {loss}")
            for k, v in reg_loss_dict.items():
                print(f"{k}: {v}")

import torch.nn.functional as F
# from torch_influence import AutogradInfluenceModule, LiSSAInfluenceModule, CGInfluenceModule
# from torch_influence import BaseObjective

# class MyObjective(BaseObjective):
#     # The name is so uninformative
#     def __init__(self, C):
#         self.C = C
    
#     def train_outputs(self, model, batch):
#         # self.train_batch = batch
#         return model(batch[0])

#     def train_loss_on_outputs(self, outputs, batch):
#         ce_loss = F.cross_entropy(outputs, batch[1])
#         return ce_loss

#     def train_regularization(self, params):
#         # training loss by default taken to be 
#         # train_loss_on_outputs + train_regularization
#         return self.C * torch.norm(params, p=2) ** 2

#     def test_loss(self, model, params, batch):
#         ce_loss = F.cross_entropy(model(batch[0]), batch[1]) 
#         return ce_loss
    
class LinearProbingModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.feature_norm = cfg.training.feature_norm
        if cfg.mode == "linear_probing_with_model":
            self.feature_model = get_model(cfg.backbone.name, skip_pool=cfg.backbone.skip_pool, pretrain_path=cfg.backbone.pretrain_path, img_dim=cfg.backbone.img_dim)[0]
            # frozen the feature model
            for param in self.feature_model.parameters():
                param.requires_grad = False

        if self.cfg.backbone.name == "resnet50":
            self.classifier = RegLog(10, 2048, use_bn=True, reinit_head=False)
        elif self.cfg.backbone.name == "resnet18":
            self.classifier = RegLog(10, 512, use_bn=True, reinit_head=False)
        self.criterion = nn.CrossEntropyLoss()
        self.acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10)
    
    def forward(self, x):
        feat = x
        # if self.cfg.mode == "linear_probing_with_model":
        #     feat = self.feature_model(x)
        # elif self.cfg.mode == "linear_probing_with_eb":
        #     feat = x
        # feat = self.feature_model.get_feature(x)
        if self.feature_norm:
            feat = feat / feat.norm(dim=1, keepdim=True)
        logit = self.classifier(feat)
        return logit

    def general_step(self, batch, batch_idx, phase="train"):
        x, y = batch
        y_hat = self(x)
        y_loss = self.criterion(y_hat, y)
        loss = y_loss
        y_acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log(f"{phase}/y_loss", y_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{phase}/y_acc", y_acc, prog_bar=True, on_step=True, on_epoch=True)
        # log lr 
        for idx, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(f"{phase}/lr/param_group={idx}", param_group["lr"], prog_bar=True, on_step=True, on_epoch=True)
        # self.log(f"{phase}/lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "test")
    
    def configure_optimizers(self):
        from back.optimizer import optimizer_config
        # optimizer = optimizer_config(self.classifier, self.cfg)
        if self.cfg.training.optimizer.type == "SGD":
            optimizer = SGD(self.classifier.parameters(), 
                            lr=self.cfg.training.optimizer.lr, 
                            momentum=self.cfg.training.optimizer.momentum, 
                            weight_decay=self.cfg.training.optimizer.wd)
        elif self.cfg.training.optimizer.type == "Adam":
            optimizer = torch.optim.Adam(self.classifier.parameters(), 
                            lr=self.cfg.training.optimizer.lr, 
                            weight_decay=self.cfg.training.optimizer.wd)

        logger.info(f"optimizer: {optimizer}")
        if self.cfg.training.scheduler.type == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg.training.scheduler.step_size, gamma=self.cfg.training.scheduler.gamma)
        elif self.cfg.training.scheduler.type == "CosineAnnealingLR":
            total_epochs = self.cfg.training.max_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs, eta_min=self.cfg.training.scheduler.eta_min)
        elif self.cfg.training.scheduler.type == "null":
            scheduler = None
        else:
            scheduler = None
        if scheduler is not None:
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        else:
            return optimizer

def get_sklearn_full_selection_model_path(cfg):
    path = f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}/selection#method=full-data_type=eb/sklearn_model={cfg.training.arch}.pt"
    return path

def sklearn_linear_probing(cfg, datamodule):
    # check using MLP
    wandb.init(project="data_pruning", entity="WANDB_ENTITY", config=OmegaConf.to_container(cfg, resolve=True))
    train_data = datamodule.train_dataset.data.detach()
    train_targets = datamodule.train_dataset.targets.detach()
    val_data = datamodule.val_dataset.data.detach()
    val_targets = datamodule.val_dataset.targets.detach()
    test_data = datamodule.test_dataset.data.detach()
    test_targets = datamodule.test_dataset.targets.detach()

    #* process the selection
    train_selection_indices = datamodule.train_selection_indices
    if isinstance(train_selection_indices, np.ndarray):
        train_selection_indices = torch.from_numpy(train_selection_indices).long()
    logger.debug(datamodule.train_selection_indices.shape)
    logger.debug(datamodule.train_selection_indices)
    train_data = train_data[train_selection_indices]
    train_targets = train_targets[train_selection_indices]
    #TODO[0]: move this to datamodule
    if cfg.training.sample_weight == "s":
        train_val_sample_weight = torch.cat([datamodule.sample_weight, torch.ones_like(val_targets)], dim=0)
    else:
        train_val_sample_weight = None
    train_val_data = torch.cat([train_data, val_data], dim=0).detach()
    train_val_targets = torch.cat([train_targets, val_targets], dim=0).detach()

    logger.info(f"train_data: {train_data.shape}")
    logger.info(f"val_data: {val_data.shape}")
    logger.info(f"train_val_data: {train_val_data.shape}")

    test_fold = [-1] * len(datamodule.train_selection_indices) + [0] * len(val_data)
    ps = PredefinedSplit(test_fold=test_fold)
    
    if cfg.training.arch == "mlp_50":
        clf = MLPClassifier(hidden_layer_sizes=(50), activation="identity")
        model = Pipeline([("scaler", StandardScaler()), ("clf", clf)], verbose=True)
        param_grid = {
            "clf": [MLPClassifier(hidden_layer_sizes=(50), activation="identity")],
            "clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            "clf__solver": ["lbfgs"],
            "clf__max_iter": [100],
        }
    else:
        model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())], verbose=True)
    
        param_grid = {
            "clf__C": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            "clf__solver": ["lbfgs"],
            #! note: I tried 1000, no difference for test acc
            "clf__max_iter": [100],
        }
    grid_search = GridSearchCV(model, param_grid, cv=ps, n_jobs=-1, verbose=1)
    grid_search.fit(train_val_data, train_val_targets)
    logger.info(grid_search.best_params_)
    logger.info(grid_search.best_score_)
    logger.info(grid_search.best_estimator_)
    best_model = grid_search.best_estimator_
    # best_model.fit(train_data, train_targets)
    y_pred = best_model.predict(test_data)
    test_acc = accuracy_score(test_targets, y_pred)
    logger.info(f"test_acc: {test_acc}")
    
    # log best hyperparameters
    train_acc = accuracy_score(train_val_targets, best_model.predict(train_val_data))
    val_acc = accuracy_score(val_targets, best_model.predict(val_data))
    
    # wandb.log({"best_params": grid_search.best_params_})
    wandb.log({"train/acc": train_acc})
    wandb.log({"val/acc": val_acc})
    wandb.log({"test/acc": test_acc})
    for k, v in grid_search.best_params_.items():
        # check if __ in k
        if "__" in k:
            wandb.log({f"best_params/{k}": v})
            
    
    # save the best model to index dir
    path = f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}/selection#{cfg.selection.str}"
    os.makedirs(path, exist_ok=True)
    best_model_path = f"{path}/sklearn_model={cfg.training.arch}.pt"
    torch.save(best_model, best_model_path)
    return best_model

def get_last_ckpt_path(cfg):
    last_ckpt_dir = f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}"
    last_linear_path = f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}/last_linear.pt"
    if cfg.pretraining is not None:
        last_linear_path = f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}/pretraining#{cfg.pretraining.str}/last_linear.pt"
        last_ckpt_dir = f"{last_ckpt_dir}/pretraining#{cfg.pretraining.str}"
    last_ckpt_path = f"{last_ckpt_dir}/last.ckpt"
    return last_ckpt_path

def table_feature_extraction(cfg):
    dataset_name = cfg.dataset.name
    from task_datasets.cardio import CardioDataset
    train_dataset = CardioDataset(split="train")
    val_dataset = CardioDataset(split="val")
    test_dataset = CardioDataset(split="test")
    # 
    
# @PipelineDecorator.component(cache=True, execution_queue="default")
def run_feature_extraction(cfg):
    if cfg.dataset.name in ["cardio"]:
        table_feature_extraction(cfg)
        return
    
    #! mode_2: feature extraction
    datamodule = DataModule(cfg)
    last_ckpt_dir = f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}"
    last_linear_path = f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}/last_linear.pt"
    # if cfg.pretraining is not None:
    if hasattr(cfg, "pretraining"):
        last_linear_path = f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}/pretraining#{cfg.pretraining.str}/last_linear.pt"
        
        last_ckpt_dir = f"{last_ckpt_dir}/pretraining#{cfg.pretraining.str}"
        last_ckpt_path = f"{last_ckpt_dir}/last.ckpt"
        last_ckpt_path = os.path.realpath(last_ckpt_path)
        ckpt_name = last_ckpt_path.split("/")[-1]
        last_ckpt_path = f"{last_ckpt_dir}/{ckpt_name}"
        # last_linear_path = f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}/pretraining#{cfg.pretraining.str}/last_linear.pt"
        if os.path.exists(last_linear_path):
            logger.info(f"last_linear_path: {last_linear_path} exists")
        ckpt = torch.load(last_ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]
        
        ckpt["state_dict"] = state_dict
        torch.save(ckpt, last_ckpt_path)
        
        module = FeatureExtractionModule.load_from_checkpoint(last_ckpt_path, cfg=cfg)
    else:
        module = FeatureExtractionModule(cfg)


    #FIXME: issue with multiple gpus
    trainer = pl.Trainer(accelerator="gpu",
                            devices=[0],
                            fast_dev_run=cfg.debug)
    module.dataset_phase = "train"
    trainer.test(module, dataloaders=[datamodule.train_dataloader()])
    module.dataset_phase = "val"
    trainer.test(module, dataloaders=[datamodule.val_dataloader()])
    module.dataset_phase = "test"
    trainer.test(module, dataloaders=[datamodule.test_dataloader()])
    # save classifier from last checkpoint
    if "clip" in cfg.backbone.name:
        pass
    elif "tinynet" in cfg.backbone.name:
        pass
    elif "vit_tiny" in cfg.backbone.name:
        pass
    else:
        classifier = module.net.net.fc
        logger.debug(f"classifier: {classifier}")
        logger.debug(f"type(classifier): {type(classifier)}")
        # only last linear
        if type(classifier) == nn.Sequential:
            last_linear = classifier[-1]
        elif type(classifier) == nn.Linear:
            last_linear = classifier
        else:
            last_linear = classifier.fc1
        with open(f"{last_linear_path}", "wb") as f:
            torch.save(last_linear, f)

# @PipelineDecorator.component(cache=True, execution_queue="default")
def run_combined_optimization(save_path, debug=False):
    import cvxpy as cp
    import os
    import torch
    import submitit
    def m_guided_opt(S, size):
        n,m = S.shape
        W = cp.Variable(n, boolean=True)
        constaints = [cp.sum(W)==size]
        obj = cp.Minimize(cp.norm(W@S,2))
        prob = cp.Problem(obj, constaints)
        prob.solve(solver=cp.CPLEX, verbose=True)
        W_optim = W.value
        return W_optim
    # just submite and save
    def do_opt(save_path, fraction):
        with open(f"{save_path}/S.pt", "rb") as f:
            S = torch.load(f)
        train_size = S.shape[0]
        W = m_guided_opt(S=S, size=(train_size * fraction))
        prune_index = (W==0)
        select_index = (W==1)
        fraction_dir = f"{save_path}/fraction={fraction}"
        os.makedirs(fraction_dir, exist_ok=True)
        # train_index only in train
        torch.save(select_index, f"{fraction_dir}/train_index.pt")
        torch.save(prune_index, f"{fraction_dir}/prune_index.pt")
    job_list = []
    # os.system(f"rm -rf ./logs/submitit/*")
    for fraction in [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]:
        if os.path.exists(f"{save_path}/fraction={fraction}/train_index.pt"):
            continue
        else:
            if fraction == 1.0:
                # save the full index
                with open(f"{save_path}/S.pt", "rb") as f:
                    S = torch.load(f)
                train_size = S.shape[0]
                train_index = torch.arange(train_size)
                fraction_dir = f"{save_path}/fraction={fraction}"
                os.makedirs(fraction_dir, exist_ok=True)
                torch.save(train_index, f"{fraction_dir}/train_index.pt")
                
            if debug:
                do_opt(save_path, fraction)
            else:
                executor = submitit.AutoExecutor(folder=f"./logs/submitit")
                executor.update_parameters(
                    timeout_min=60*4, 
                    slurm_partition="veitch-contrib",
                    name=f"fraction={fraction}",
                    cpus_per_task=32, 
                    gpus_per_node=1, 
                    nodes=1, 
                    mem_gb=400)
                job = executor.submit(do_opt, save_path, fraction)
                job_list.append(job)

def run_pretraining(cfg, datamodule, name):
    import os, sys, logging
    REMOTE_LABROOT = os.getenv("REMOTE_LABROOT")
    sys.path.append(REMOTE_LABROOT)
    import torch
    from models.litclassifier import LitClassifier
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    pl.seed_everything(cfg.training.seed)
    save_dir = f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}/pretraining#{cfg.training.str}"
    max_epochs = 200
    module = LitClassifier(cfg)
    pl_checkpoint = ModelCheckpoint(dirpath=save_dir,
                                    monitor="val/acc",
                                    save_last=True)
    # check if there is a checkpoint
    if os.path.exists(f"{save_dir}/last.ckpt"):
        logger.info(f"loading from {save_dir}/last.ckpt")
        module = LitClassifier.load_from_checkpoint(f"{save_dir}/last.ckpt", cfg=cfg)
    else:
        print("training from scratch!")
        logger.info(f"training from scratch")
    # check
    logger.info(f"save_dir: {save_dir}")
    wandb_logger = WandbLogger(project="data_pruning",
                                entity="WANDB_ENTITY", 
                                name=name)
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    trainer = pl.Trainer(default_root_dir=name,
                            max_epochs=max_epochs,
                            accelerator="auto",
                            callbacks=[pl_checkpoint],
                            logger=wandb_logger)
    trainer.fit(module, datamodule)
    trainer.test(module, datamodule)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(module.net.state_dict(), f"{save_dir}/model.pt")
    
@hydra.main(config_path="configs", config_name="default", version_base="1.3.0")
def main(cfg):
    #* task setup
    dataset_str = cfg.dataset.str if hasattr(cfg.dataset, "str") else cfg.dataset._target_.split(".")[-1]
    name = f"mode={cfg.mode}/backbone#{cfg.backbone.str}/dataset#{dataset_str}"
    pl.seed_everything(cfg.seed)
    logger.info(OmegaConf.to_yaml(cfg))
    datamodule = DataModule(cfg)
    log_dict = datamodule.setup(stage=None)
    #! mode_1 or mode_5: model_training (optional)
    #modified from 
    if cfg.mode == "pretraining":
        cfg.dataset.seed = cfg.seed
        cfg.training.seed = cfg.seed
        run_pretraining(cfg, datamodule, name)
    
    #! mode_2: feature extraction
    elif cfg.mode == "feature_extraction":
        run_feature_extraction(cfg)

    #! mode_3 or mode_6: linear probing
    elif cfg.mode == "linear_probing":
        sklearn_linear_probing(cfg, datamodule)
        if cfg.training.arch != "linear":
            # save the last representation
            get_sklearn_full_selection_model_path(cfg)
            full_sklearn_model_path = get_sklearn_full_selection_model_path(cfg)
            pipeline = torch.load(full_sklearn_model_path)
            clf = pipeline.named_steps["clf"]
            input_weights = clf.coefs_
            input_biass = clf.intercepts_
            
            linear1 = nn.Linear(2048, 50)
            linear2 = nn.Linear(50, 10)
            linear1.weight.data = torch.from_numpy(input_weights[0].T).float()
            linear1.bias.data = torch.from_numpy(input_biass[0]).float()
            linear2.weight.data = torch.from_numpy(input_weights[1].T).float()
            linear2.bias.data = torch.from_numpy(input_biass[1]).float()
            
            ori_dict = {
                "train": {
                    "X": datamodule.train_dataset.data,
                    "Y": datamodule.train_dataset.targets,
                },
                "val": {
                    "X": datamodule.val_dataset.data,
                    "Y": datamodule.val_dataset.targets,
                },
                "test": {
                    "X": datamodule.test_dataset.data,
                    "Y": datamodule.test_dataset.targets,
                },
            }
            new_dict = {}
            for k, v in ori_dict.items():
                X = v["X"]
                Y = v["Y"]
                X = linear1(X)
                new_dict[k] = {
                    "X": X,
                    "Y": Y,
                }
            
            #! save the features
            # get_sklearn_full_selection_model_path
            feature_path = f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}/rep#{cfg.training.arch}"
            for phase, data in new_dict.items():
                phase_path = f"{feature_path}/{phase}"
                os.makedirs(phase_path, exist_ok=True)
                X = data["X"]
                Y = data["Y"]
                torch.save(X, f"{phase_path}/X.pt")
                torch.save(Y, f"{phase_path}/Y.pt")

    #! mode_4: selection
    elif cfg.mode == "selection":
        if cfg.selection.method == "full":
            train_dataset = datamodule.train_dataset
            index = torch.arange(len(train_dataset))
            # save the index
            dir_dict = {
                "root": "cached_datasets",
                "backbone": cfg.backbone.str,
                "dataset": cfg.dataset.str,
                "selection": cfg.selection.str,
            }
            dir = os.path.join(*[f"{k}={v}" for k, v in dir_dict.items() if k != "root"])
            # add the root
            dir = os.path.join(dir_dict["root"], dir)
            os.makedirs(dir, exist_ok=True)
            with open(os.path.join(dir, "dir_dict.yaml"), "w") as f:
                yaml.dump(dir_dict, f)
            torch.save(index, os.path.join(dir, "train_index.pt"))

        elif cfg.selection.method in ["uniform", "forgetting"]:
            train_dataset = datamodule.train_dataset
            selection_args = {}
            selection_method = methods.__dict__[cfg.selection.method.capitalize()](
                                        dst_train=train_dataset,
                                        args=cfg.selection,
                                        fraction=cfg.selection.fraction,
                                        random_seed=cfg.selection.seed,
                                        **selection_args)
            global_seed = torch.initial_seed()
            pl.seed_everything(cfg.selection.seed)
            index = selection_method.select()
            pl.seed_everything(global_seed)
            dir_dict = {
                "root": "cached_datasets",
                "backbone": cfg.backbone.str,
                "dataset": cfg.dataset.str,
                "selection": cfg.selection.str,
            }
            dir = os.path.join(*[f"{k}#{v}" for k, v in dir_dict.items() if k != "root"])
            dir = f"{dir}/fraction={cfg.selection.fraction}"
            # add the root
            dir = os.path.join(dir_dict["root"], dir)
            os.makedirs(dir, exist_ok=True)
            with open(os.path.join(dir, "dir_dict.yaml"), "w") as f:
                yaml.dump(dir_dict, f)
            torch.save(index, os.path.join(dir, "train_index.pt"))

        elif cfg.selection.method == "covif":
            # already handled in the datamodule
            train_X = datamodule.train_dataset.data
            train_Y = datamodule.train_dataset.targets
            val_X = datamodule.val_dataset.data
            val_Y = datamodule.val_dataset.targets
            test_X = datamodule.test_dataset.data
            test_Y = datamodule.test_dataset.targets
            
            # cov selection
            covif_list = []
            # Cov = (1/n) * X.T @ X
            # Cov_{\x} = (1/n) * X_{\x + eps}.T @ X_{\x + eps}
            # lim_{eps -> 0} Cov_{\x} 
            # = lim_{eps -> 0} (1/n) * (X_{\x} + eps).T @ (X_{\x} + eps)
            # = lim_{eps -> 0} (1/n) * (X_{\x}.T @ X_{\x} + eps * X_{\x} + eps * X_{\x} + eps^2)
            # = lim_{eps -> 0} (1/n) * (X_{\x}.T @ X_{\x} + 2 * eps * X_{\x} + eps^2)
                
            mean = torch.mean(train_X, dim=0)
            # n = train_X.shape[0]
            for i in len(train_X):
                x = train_X[i]
                cov = torch.cov(train_X)
                # sample point cov
                x_cov = (x - mean).T @ (x - mean)
                covif = - (2/n) * torch.norm(x_cov - cov)
                covif_list.append(covif)
            covif_list = torch.stack(covif_list)
            with open(f"{save_path}/covif.pt", "wb") as f:
                torch.save(covif_list, f)
            
            #! selection
            # cov_score
            
            
            # cov_combine
            
            
        elif cfg.selection.method == "influence_function":
            # check linear probing exists
            # FIXME: wrap it
            # If the training.classifier is mlp_50, we can directly use the model and feature
            if cfg.backbone.classifier == "mlp_50":
                # model = torch.load(f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}/selection#{cfg.selection.str}/.pt")
                last_linear = torch.load(f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}/last_linear.pt")

                features = datamodule.train_dataset.data
                labels = datamodule.train_dataset.targets
            else:
                # If the training.classifeir is linear, we need to linear probing
                full_sklearn_model_path = get_sklearn_full_selection_model_path(cfg)
                pipeline = torch.load(full_sklearn_model_path)
                clf = pipeline.named_steps["clf"]
                input_weights = clf.coefs_
                input_biass = clf.intercepts_
                
                linear1 = nn.Linear(2048, 50)
                linear2 = nn.Linear(50, 10)
                linear1.weight.data = torch.from_numpy(input_weights[0].T).float()
                linear1.bias.data = torch.from_numpy(input_biass[0]).float()
                linear2.weight.data = torch.from_numpy(input_weights[1].T).float()
                linear2.bias.data = torch.from_numpy(input_biass[1]).float()
                
                linear_feature_extractor = linear1
                
                features = datamodule.train_dataset.data
                features = linear1(features)
                labels = datamodule.train_dataset.targets
                classifier = linear2
            
            # #FIXME: hard code
            # classifier = nn.Linear(input_dim, output_dim)
            # classifier.weight.data = torch.from_numpy(clf.coef_).float()
            # classifier.bias.data = torch.from_numpy(clf.intercept_).float()
            # save classifier
            save_path = f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}/selection#{cfg.selection.str}"
            os.makedirs(save_path, exist_ok=True)
            with open(f"{save_path}/last_linear.pt", "wb") as f:
                torch.save(last_linear, f)
                
            # process the data here
            all_data = (features, labels)
            with open(f"{save_path}/all_data.pt", "wb") as f:
                torch.save(all_data, f)
            
            IF_config = cfg.selection.IF
            def submit_influence_score():
                if IF_config.method == "LiSSA":
                    train_dataloader = datamodule.train_dataloader()
                    test_dataloader = datamodule.test_dataloader()
                    # os.system(f"rm -rf ./logs/submitit/*")
                    executor = submitit.AutoExecutor(folder=f"./logs/submitit")
                    executor.update_parameters(
                        timeout_min=60*4, 
                        slurm_partition="veitch-contrib",
                        cpus_per_task=1, 
                        gpus_per_node=1, 
                        nodes=1, 
                        mem_gb=32)
                    # submitit
                    batch_size = 500
                    num_batches = len(features) // batch_size
                    job_list = []
                    for batch_idx in range(num_batches):
                        if os.path.exists(f"{save_path}/influence_score-batch_idx={batch_idx}-batch_size={batch_size}.pt"):
                            print(f"skip batch_idx={batch_idx}")
                            continue
                        # input(f"{save_path}/influence_score-batch_idx={batch_idx}-batch_size={batch_size}.pt cannot be found, press enter to continue")
                        if cfg.debug:
                            get_influence_score_batch(model_path=f"{save_path}/last_linear.pt",
                                                    all_data_path=f"{save_path}/all_data.pt",
                                                    batch_idx=batch_idx, 
                                                    batch_size=batch_size,
                                                    save_path=save_path)
                        else:
                            executor.update_parameters(
                                name=f"{save_path}/batch_idx={batch_idx}-batch_size={batch_size}")
                            job = executor.submit(get_influence_score_batch, 
                                            model_path=f"{save_path}/last_linear.pt",
                                            all_data_path=f"{save_path}/all_data.pt",
                                            batch_idx=batch_idx, 
                                            batch_size=batch_size,
                                            save_path=save_path)
                            job_list.append(job)
                    for job in job_list:
                        job.result()
                    influence_score_list = []
                    for batch_idx in tqdm(range(num_batches)):
                        try:
                            score = torch.load(f"{save_path}/influence_score-batch_idx={batch_idx}-batch_size={batch_size}.pt")
                        except:
                            print(f"skip batch_idx={batch_idx}")
                        influence_score_list.append(score)
                    influence_score_list = np.concatenate(influence_score_list)
                    influence_score_list = torch.from_numpy(influence_score_list)
                    with open(f"{save_path}/influence_score.pt", "wb") as f:
                        torch.save(influence_score_list, f)
                    
                    dataset_size = influence_score_list.shape[0]
                    S = (-1/dataset_size) * influence_score_list
                    torch.save(S, f"{save_path}/S.pt")
                    return S
                else:
                    device = "cuda"
                    x = train_dataloader.dataset.data.to(device)
                    y = train_dataloader.dataset.targets.to(device)
                    model = classifier.to(device)
                    num_param = sum(p.numel() for p in model.parameters())
                    names = list(n for n, _ in model.named_parameters())
                    def loss(params):
                        y_hat = torch.func.functional_call(model, {n: p for n, p in zip(names, params)}, x)
                        return F.cross_entropy(y_hat, y) + 0.01 * torch.norm(params[-1], p=2) ** 2
                    
                    def loss_with_input(params, x, y):
                        y_hat = torch.func.functional_call(model, {n: p for n, p in zip(names, params)}, x)
                        return F.cross_entropy(y_hat, y) + 0.01 * torch.norm(params[-1], p=2) ** 2
                    H_inv_path = f"{save_path}/H_inv.pt"
                    
                    if os.path.exists(H_inv_path):
                        H_inv = torch.load(H_inv_path, map_location=device)
                    else:
                        hessian_func = torch.func.hessian(loss)
                        H = hessian_func(tuple(model.parameters()))
                        H = torch.cat([torch.cat([e.flatten() for e in Hpart]) for Hpart in H]) # flatten
                        H = H.reshape(num_param, num_param)
                        print(H.shape)
                        os.makedirs(save_path, exist_ok=True)
                        with open(f"{save_path}/H.pt", "wb") as f:
                            torch.save(H, f)
                        H_inv = torch.inverse(H)
                        with open(f"{save_path}/H_inv.pt", "wb") as f:
                            torch.save(H_inv, f)

                    # calculate the hessian product
                    ihp_list = []
                    from torch.utils.data import TensorDataset
                    train_dataloader = torch.utils.data.DataLoader(TensorDataset(train_dataloader.dataset.data, train_dataloader.dataset.targets), batch_size=1)
                    model.train()
                    params = [p for n, p in model.named_parameters()]
                    params = torch.cat([e.flatten() for e in params])
                    l2_reg = 0.01 * torch.norm(params, p=2) ** 2
                    for x, y in tqdm(train_dataloader):
                        x = torch.tensor(x).to(device)
                        y = torch.Tensor(y).to(device)
                        def loss_with_input(model, x, y):
                            y_hat = model(x)
                            ce_loss = F.cross_entropy(y_hat, y)
                            return ce_loss + l2_reg
                            # return F.cross_entropy(y_hat, y) + 0.01 * torch.norm([p for n, p in model.named_parameters()], p=2) ** 2
                        grad = torch.autograd.grad(loss_with_input(model, x, y), model.parameters(), create_graph=True)
                        grad = torch.cat([e.flatten() for e in grad])
                        ihp = H_inv @ grad
                        ihp_list.append(ihp)
                    ihp_list = torch.stack(ihp_list)
                    with open(f"{save_path}/ihp_list.pt", "wb") as f:
                        torch.save(ihp_list, f)
            # return
            if os.path.exists(f"{save_path}/S.pt"):
                S = torch.load(f"{save_path}/S.pt")
            else:
                S = submit_influence_score()
            # return
            run_combined_optimization(save_path=save_path, debug=cfg.debug)

            # do the selection
            # #TODO: fix hard code
            # get_influence_score_batch(model_path=f"{save_path}/classifier.pt",
            #                             all_data_path=f"{save_path}/all_data.pt",
            #                             batch_idx=0, 
            #                             batch_size=1,
            #                             save_path=save_path)
            # batch_size = 500
            # num_batches = len(features) // batch_size + 1
            # for batch_idx in range(num_batches):
            #     # check if the file exists
            #     if os.path.exists(f"{save_path}/influence_score-batch_idx={batch_idx}-batch_size={batch_size}.npy"):
            #         continue
            #     else:
            #         # submitit
            #         get_influence_score_batch(model_path=f"{save_path}/classifier.pt",
            #                             all_data_path=f"{save_path}/all_data.pt",
            #                             batch_idx=batch_idx, 
            #                             batch_size=batch_size,
            #                             save_path=save_path)
            # merge the influence score
            # print()
            # influence_score_list = []
            # for batch_idx in range(num_batches):
            #     score = np.load(f"{save_path}/influence_score-batch_idx={batch_idx}-batch_size={batch_size}.npy")
            #     print(score)
                # influence_score_list.append(np.load(f"{save_path}/influence_score-batch_idx={batch_idx}-batch_size={batch_size}.npy"))
            # influence_score_list = np.concatenate(influence_score_list)
            # np.save(f"{save_path}/influence_score.npy", influence_score_list)

        elif cfg.selection.method == "optimization":
            #! TODO
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = RegSelection(10, 2048, use_bn=True, reinit_head=False).to(device)
            X_train = torch.load(f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}/phase=train/X.pt")
            X_val = torch.load(f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}/phase=val/X.pt")
            y_train = torch.load(f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}/phase=train/Y.pt")
            X_train = X_train.to(device)
            X_val = X_val.to(device)
            y_train = y_train.to(device)
            model.fit(X_train, X_val, y_train)
    

    elif cfg.mode in ["linear_probing_with_model", "linear_probing_with_eb", "linear_probing_selection"]:
        callbacks = []
        tags = []
        name = f"mode={cfg.mode}/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}/{cfg.selection.str}/training={cfg.training.str}"
        # log the cfg
        wandb_logger = WandbLogger(entity="WANDB_ENTITY",
                                    project="data_pruning",
                                    name=name,
                                    dir="output/",
                                    tags=tags)
        dict_cfg = OmegaConf.to_container(cfg, resolve=True)
        wandb_logger.log_hyperparams(dict_cfg)

        if "sklearn" in cfg.training.str:
            sklearn_linear_probing(cfg, datamodule)
        else:
            # self training
            model_checkpoint = ModelCheckpoint(
                monitor="val/y_acc",
                dirpath="checkpoints",
                mode="max",
                save_top_k=1,
                save_last=True,
                filename="{epoch}-{val/y_acc:.4f}",
            )
            callbacks += [model_checkpoint]
            module = LinearProbingModule(cfg)
            trainer = pl.Trainer(
                                    accelerator="gpu", 
                                    fast_dev_run=cfg.debug,
                                    callbacks=callbacks,
                                    logger=wandb_logger,
                                    max_epochs=cfg.training.max_epochs,
                                )

            trainer.fit(module, datamodule=datamodule)
            trainer.test(module, datamodule=datamodule)

    wandb.finish()


if __name__ == "__main__":
    main()