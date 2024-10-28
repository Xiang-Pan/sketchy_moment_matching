from omegaconf import OmegaConf
import pickle as pkl
from collections import defaultdict
import torch
import os
import numpy as np
import os
import copy
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import hydra
import sklearn
import yaml
import torchvision
import pytorch_lightning as pl
import itertools
from scipy.optimize import minimize
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from utils.data_utils import load_eb_dataset_cfg, get_raw_dataset_splits, get_output_dir, get_output_dir_without_fraction, get_grad_dir
from utils.sampling_utils import select_by_prob

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, ConstantLR
from hydra.utils import instantiate

from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from models.litclassifier import ResNet
import json
from linear_probe import test_selection
import json
from utils.minio_utils import get_minio_client, exists_on_minio
import sys
import boto3

import wandb
# wandb.require("core")

from joblib import Memory
cache_location = "./cache"
memory = Memory(cache_location, verbose=1)

import logging
logging.basicConfig(level=logging.INFO, filename=f"logs/{__name__}.log")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from lightning.pytorch import LightningModule, Trainer, LightningDataModule

def migrate_state_dict(state_dict, old_key, new_key, keep_old_key=False):
    """_summary_
    If new_key is None, the old_key will be removed

    Args:
        state_dict (_type_): _description_
        old_key (_type_): _description_
        new_key (_type_): _description_
        keep_old_key (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if old_key in k:
            if new_key is None:
                continue
            new_k = k.replace(old_key, new_key)
            if keep_old_key:
                new_state_dict[k] = v
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict


class sModule(nn.Module):
    def __init__(self, m, d):
        super(sModule, self).__init__()
        self.s = nn.Parameter(torch.ones((d)).float() * m / d)

    def forward(self, x):
        return self.s @ x

def get_sample_wise_loss(model, X, Y, task="classification"):
    if task == "classification":
        train_y_logit = model.predict_proba(X)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        sample_wise_loss = loss_fn(torch.tensor(train_y_logit), torch.tensor(Y, dtype=torch.long)).cpu().detach()
    else:
        train_y_pred = model.predict(X)
        loss_fn = nn.MSELoss(reduction="none")
        sample_wise_loss = loss_fn(torch.tensor(train_y_pred), torch.tensor(Y)).cpu().detach()
    return sample_wise_loss

def get_last_ckpt_path(cfg):
    last_ckpt_dir = f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}"
    last_linear_path = f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}/last_linear.pt"
    if hasattr(cfg, "pretraining"):
        last_linear_path = f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}/pretraining#{cfg.pretraining.str}/last_linear.pt"
        last_ckpt_dir = f"{last_ckpt_dir}/pretraining#{cfg.pretraining.str}"
    last_ckpt_path = f"{last_ckpt_dir}/last.ckpt"
    return last_ckpt_path

def get_last_linear_path(cfg):
    last_linear_path = f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}/last_linear.pt"
    if cfg.pretraining is not None:
        last_linear_path = f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}/pretraining#{cfg.pretraining.str}/last_linear.pt"
    return last_linear_path

def make_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach()
        x = x.numpy()
    assert isinstance(x, np.ndarray)
    # x = np.float64(x)
    return x
# from lightning import LightningModule, Trainer, LightningDataModule
# import lightning as L

# def check_cov_covering(Phi, selected_idxes):
#     Phi = Phi.cpu()
#     selected_Phi = Phi[selected_idxes]
#     cov = selected_Phi.T @ selected_Phi
#     cov_inv = torch.inverse(cov)
#     cov_covering = torch.trace(cov @ cov_inv)
#     return cov_covering

# import neuromancer as nm

# class MyBoundsNM(nm.Module):
#     def __init__(self, d, p, c, eigenvectors=None, eigenvalues=None, X=None, s_threshold=1e-1, fraction=0.1):
#         super(MyBoundsNM, self).__init__()
#         self.d = d
#         self.p = p
#         self.m = int(d * fraction)
#         self.c = c
#         self.s_threshold = s_threshold
#         self.eigenvectors = eigenvectors
#         self.eigenvalues = eigenvalues
#         self.X = X

#         # Defining nodes in the NeuroMANCER graph
#         self.s = nm.Parameter(torch.ones((d)) * self.m / d)
#         self.gamma = nm.Parameter(torch.ones((p)))

#         # Additional operations might be defined as separate nodes or as part of the forward method

#     def forward(self):
#         # Define the computational graph for the forward pass

#         # Normalize s and compute S
#         s_normalized = self.s / nm.sum(self.s) * self.m
#         S = nm.diag(s_normalized)

#         # Compute Covariances
#         X_centered = self.X - nm.mean(self.X, dim=0)
#         Cov = nm.matmul(X_centered.T, X_centered)

#         SX = nm.matmul(S, self.X)
#         SX_centered = SX - nm.mean(SX, dim=0)
#         Cov_S = nm.matmul(SX_centered.T, SX_centered)

#         # Quadratic form and losses
#         quadratic_form = nm.matmul(nm.matmul(self.eigenvectors.T, Cov_S), self.eigenvectors)
#         term_1 = quadratic_form - self.gamma * self.eigenvalues * nm.norm(S, p=1)
#         cov_loss = nm.norm(term_1, p=2) / self.d
#         sparse_loss = nm.norm(s_normalized, p=0) / self.d

#         loss = cov_loss + sparse_loss

#         # Additional computations for cov_gap and random_cov_gap
#         # ...
#         return loss
#         # Returning the loss and other metrics
#         # return loss, {
#         #     "cov_loss": cov_loss,
#         #     "sparse_loss": sparse_loss,
#         #     # Include other metrics here
#         # }




# def cvx(X, c, U, Sigma):
#     X = X[:100,:]
#     # make sure everything is numpy
#     import torch
#     X = X.cpu().detach().numpy() if torch.is_tensor(X) else X
#     U = U.cpu().detach().numpy() if torch.is_tensor(U) else U
#     Sigma = Sigma.cpu().detach().numpy() if torch.is_tensor(Sigma) else Sigma

#     import cvxpy as cp
#     import torch
#     from cvxpylayers.torch import CvxpyLayer

#     d, p = X.shape
#     S = cp.Variable((d, d), diag=True)
#     gamma = cp.Variable(p)
#     constraints = [S >= 0, S <= 1, gamma >= 1 / c, gamma <= c]
#     SX = S @ X
#     Cov_S = X.T @ S @ X
#     target = U.T @ Cov_S @ U - gamma * Sigma
#     # cp.norm(S, p=1)
#     target = cp.norm(target, p="fro")
#     objective = cp.Minimize(target)
#     problem = cp.Problem(objective, constraints)
#     assert problem.is_dpp()
#     problem.solve(verbose=True)
#     # cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
#     # A_tch = torch.randn(m, n, requires_grad=True)
#     # b_tch = torch.randn(m, requires_grad=True)

#     # solve the problem
#     # solution, = cvxpylayer(A_tch, b_tch)

#     # compute the gradient of the sum of the solution with respect to A, b
#     # solution.sum().backward()

#! Hoang
# alpha: 2
# batchsize: 128
# c: 1.0
# cached_feature: null
# dataset: cifar10
# epochs: 0
# lr: 0.001
# m: 1000
# output_dir: outputs/2nd_phase/seed=0/lr=0.001/retrain_lr=0.01/pgd_lr=0.01/dataset=cifar10/cached_feature=None/m=1000/batchsize=128/epochs=0/random_prune=False/c=1.0/reg_tradeoff=0.1/alpha=2
# pgd_lr: 0.01
# random_prune: false
# reg_tradeoff: 0.1
# retrain_lr: 0.01
# seed: 0

def select_eb_random(dataset_dict, m, seed, split="train", class_conditioned=False):
    if isinstance(m, torch.Tensor):
        m = m.item()
    if isinstance(m, float):
        m = int(m * dataset_dict[split]["X"].shape[0])
    elif isinstance(m, int):
        pass
    else:
        raise ValueError("m should be either int or float")
    logger.info(f"m: {m}")
    if class_conditioned:
        train_Y = dataset_dict[split]["Y"]
        unique_classes = torch.unique(train_Y)
        logger.debug("unique_classes: %s", unique_classes)
        idxes = []
        for c in unique_classes:
            idxes_c = torch.where(train_Y == c)[0]
            idxes_c = idxes_c[torch.randperm(len(idxes_c))[:m // len(unique_classes)]]
            idxes.append(idxes_c)
        idxes = torch.cat(idxes)
        idxes = idxes.cpu().detach().numpy()
    else:
        np.random.seed(seed)
        idxes = np.random.choice(dataset_dict[split]["X"].shape[0], size=m, replace=False)
    return idxes

def test_train_from_scratch(idxes, cfg):
    idxes = torch.from_numpy(idxes).long()
    from torchvision import transforms
    lenet_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset, val_dataset, test_dataset = get_raw_dataset_splits(cfg, transform=lenet_transform)

    train_dataset.transform = lenet_transform
    val_dataset.transform = lenet_transform
    test_dataset.transform = lenet_transform

    train_cfg = cfg.train_from_scratch
    from finetune import get_model
    fraction = cfg.selection.fraction
    m = int(len(train_dataset) * fraction)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train_from_scratch.batch_size, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.train_from_scratch.batch_size, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.train_from_scratch.batch_size, shuffle=False, num_workers=1)
    from pytorch_lightning import Trainer, LightningDataModule
    class MyDataModule(LightningDataModule):
        def __init__(self, train_dataloader, val_dataloader, test_dataloader):
            super().__init__()
            self._train_dataloader = train_dataloader
            self._val_dataloader = val_dataloader
            self._test_dataloader = test_dataloader
        def train_dataloader(self):
            return self._train_dataloader
        def val_dataloader(self):
            return self._val_dataloader
        def test_dataloader(self):
            return self._test_dataloader

    class MyLitModule(LightningModule):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.model = get_model()
        def forward(self, x):
            return self.model(x)
        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self.model(x)
            loss = F.cross_entropy(logits, y)
            return loss
        def configure_optimizers(self):
            return torch.optim.Adam(self.model.parameters(), lr=self.cfg.train_from_scratch.optimizer.lr)
    my_litmodule = MyLitModule(cfg)
    datamodule = MyDataModule(train_dataloader, val_dataloader, test_dataloader)
    from pytorch_lightning.loggers import WandbLogger
    wandb_logger = WandbLogger()
    trainer = Trainer(gpus=1, max_epochs=cfg.train_from_scratch.max_epochs, logger=wandb_logger)
    trainer.fit(my_litmodule, datamodule)


def load_default_cfg():
    with open("configs/default.yaml", "r") as f:
        cfg = OmegaConf.load(f)
    return cfg

def check_finished(hash_dir: str, check_summary: bool = False):
    if not os.path.exists(f"{hash_dir}/c_unconditioned_idxes.pt"):
        logger.info(f"{hash_dir}/c_unconditioned_idxes.pt not found")
        return False
    if not os.path.exists(f"{hash_dir}/c_conditioned_idxes.pt"):
        logger.info(f"{hash_dir}/c_conditioned_idxes.pt not found")
        return False
    if check_summary:
    # if not os.path.exists(f"{hash_dir}/s.pt"):
    #     logger.info(f"{hash_dir}/s.pt not found")
    #     return False
    # Key metrics to check in the summary JSON file
        key_list = [
            "class_unconditioned/use_weights=False/test_as_val=True/tune=False/test/acc"
            "class_unconditioned/use_weights=False/test_as_val=True/tune=False/test/f1"
        ]
        summary_file = f"{hash_dir}/wandb/latest-run/files/wandb-summary.json"
        minio_client = get_minio_client()

        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                summary = json.load(f)
            if all([k in summary for k in key_list]):
                return True
            print(f"{summary_file} does not contain all required keys.")
        elif exists_on_minio(minio_client, bucket_name="labs", object_name=summary_file):
            # Download the summary file to a local temporary file for processing
            local_temp_file = f"temp_summary.json"
            download_file(minio_client, summary_file, local_temp_file)
            # Read and process the JSON file
            with open(local_temp_file, "r") as f:
                summary = json.load(f)
            os.remove(local_temp_file)  # Clean up the local temporary file
            # Check if all required keys are in the summary
            if all([k in summary for k in key_list]):
                return True
            else:
                missing_keys = set(key_list) - set(summary.keys())
                print(f"{summary_file} does not contain all required keys.")
                print(f"Missing keys: {missing_keys}")
                return False
        else:
            print(f"{summary_file} does not exist in remote storage.")
            return False
    return True


wandb_dir = None


@hydra.main(
    config_path="configs",  # path to the directory where config.yaml is located
    config_name="default",  # name of the config file (without the .yaml extension)
    version_base="1.3.0",   # version of the code
)
def main(cfg):
    from utils.hash_utils import get_cfg_hash, get_cfg_hash_without_fraction
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_hash = get_cfg_hash(cfg_dict)


    global wandb_dir
    wandb_dir = f"outputs/selection/{cfg_hash}"
    output_dir = wandb_dir


    #TODO: check function here
    if os.path.exists(wandb_dir) and not cfg.overwrite:
        logger.info(f"{wandb_dir}")
        if check_finished(hash_dir=wandb_dir):
            logger.info(f"{wandb_dir} finished")
            # print wandb_summary
            wandb_summary = f"{wandb_dir}/wandb/latest-run/files/wandb-summary.json"
            with open(wandb_summary, "r") as f:
                summary = json.load(f)
            json_formatted_str = json.dumps(summary, indent=2)
            logger.info("\n"+ json_formatted_str)
            return
        else:
            logger.info(f"{wandb_dir} not finished, rerun")

    if cfg.dataset.name == "utk":
        task = "regression"
    else:
        task = "classification"
    logger.info("task: %s", task)
    os.makedirs(f"outputs/selection/{cfg_hash}", exist_ok=True)

    pl.seed_everything(cfg.seed)
    method = cfg.selection.method
    if method == "leverage_score":
        if hasattr(cfg.selection, "target"):
            method = "leverage_score_change"
            logger.info("method: leverage_score_change")
        else:
            method = "leverage_score"
            logger.info("method: leverage_score")

    mode = cfg.mode
    debug = cfg.debug

    if method == "cov_cvx":
        from methods.cov_opt import cov_cvx_run
        cov_cvx_run(cfg)
        return
    elif method in ["cov", "cov_ntk", "cov_ntk_perclass"]:
        from methods.cov_opt import cov_run
        cov_run(cfg)
        return
    elif method in ["SkMMv2", "SkMMv2_1"]:
        from methods.skmmv2 import skmmv2_run
        skmmv2_run(cfg)
        return
    else:
        logger.warning(f"method: {method} not implemented, be sure to check the code")

    wandb.init(
                project="data_pruning-selection",
                entity="WANDB_ENTITY",
                name=cfg_hash,
                config=cfg_dict,
                mode="online" if not cfg.debug else "disabled",
                tags=["0.09-debug"],
                dir=f"outputs/selection/{cfg_hash}"
            )
    train_dataset, val_dataset, test_dataset = get_raw_dataset_splits(cfg)
    logger.info("train_dataset: %s", len(train_dataset))
    logger.info("val_dataset: %s", len(val_dataset))
    logger.info("test_dataset: %s", len(test_dataset))

    if method == "leverage_score" and cfg.selection.space == "gradient":
        logger.info("leverage_score")
        # check the space:
        max_epochs = 1
        sample_wise_loss = -1
        eb_dataset_dict = load_eb_dataset_cfg(cfg, device="cuda")
        is_pretrained = hasattr(cfg, "pretraining")
        if is_pretrained:
            unpertrained_cfg = copy.deepcopy(cfg)
            del unpertrained_cfg.pretraining
            unpretrained_eb_dataset_dict = load_eb_dataset_cfg(unpertrained_cfg, device="cuda")
        else:
            unpretrained_eb_dataset_dict = None
        for epoch in range(max_epochs):
            leverage_score_model = LeverageScoreSelection(cfg, task=task)
            train_dataset, val_dataset, test_dataset = get_raw_dataset_splits(cfg)
            #! loading model
            if epoch == 0:
                train_X = eb_dataset_dict["train"]["X"].cpu().detach().numpy()
                train_Y = eb_dataset_dict["train"]["Y"].cpu().detach().numpy().astype(int)
                is_pretrained = hasattr(cfg, "pretraining")
                if cfg.selection.source == "pretrained":
                    #! Is there any better way?
                    # random
                    linear = nn.Linear(train_X.shape[1], cfg.dataset.num_classes)
                    linear_probing_model = LogisticRegression(max_iter=1).fit(train_X, train_Y)
                    linear_probing_model.coef_ = linear.weight.data.cpu().detach().numpy()
                    linear_probing_model.intercept_ = linear.bias.data.cpu().detach().numpy()
                    logger.info("pretrained")
                    leverage_score_model.set_classifier_from_sklearn(linear_probing_model)
                elif cfg.selection.source == "linear_trained":
                    linear_probing_model = test_eb(train_dataset, np.arange(len(train_dataset)), weights=None, use_weights=False, name="full", use_mlp=False, task=task, dataset_name=cfg.dataset.name)
                    leverage_score_model.set_classifier_from_sklearn(linear_probing_model)
                elif cfg.selection.source == "finetuned":
                    leverage_score_model.load_finetuned()
                    if type(leverage_score_model.net.net.fc) == nn.Linear:
                        fc = leverage_score_model.net.net.fc
                    else:
                        fc = leverage_score_model.net.net.fc.fc1
                    linear_probing_model = LogisticRegression(max_iter=1).fit(train_X, train_Y)
                    linear_probing_model.coef_ = fc.weight.data.cpu().detach().numpy()
                    linear_probing_model.intercept_ = fc.bias.data.cpu().detach().numpy()
                logger.info(f"source: {cfg.selection.source}")
                sample_wise_loss = get_sample_wise_loss(linear_probing_model, train_X, train_Y, task=task)

            logger.info(f"classifier_only: {cfg.selection.classifier_only}")

            leverage_score_model = leverage_score_model.cuda()

            #! selection part
            if cfg.selection.classifier_only:
                G = leverage_score_model.get_G(
                    dataset=train_dataset,
                    classifier_only=cfg.selection.classifier_only,
                    k=cfg.selection.k)
            else:
                # G = leverage_score_model.get_G(dataset=train_dataset,
                #     classifier_only=cfg.selection.classifier_only,
                #     k=cfg.selection.k,
                #     source=cfg.selection.source)
                #?: understand why "auto" is faster, but without
                #! get raw gradient
                trainer = pl.Trainer(max_epochs=1, accelerator="auto", strategy="ddp", gpus=4)
                batch_size = 40
                dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
                trainer.validate(leverage_score_model, dataloader)
                #! get sketching
                # for batch_idx in range(31):
                    # for device_id in range(4):
                def sketching(batch_idx, device_id, k=cfg.selection.k):
                    per_sample_grads = torch.load(f"{leverage_score_model.output_dir}/per_sample_gradient_batch_idx_{batch_idx}_device_{device_id}.pt").cuda()
                    S = torch.randn((per_sample_grads.shape[1], cfg.selection.k)).normal_(mean=0, std=(1/cfg.selection.k)**0.5).to("cuda")
                    sketched_per_sample_grads = (per_sample_grads @ S).cpu().detach()
                    torch.cuda.empty_cache()
                    torch.save(sketched_per_sample_grads, f"{leverage_score_model.output_dir}/sketched_grads_batch_idx={batch_idx}_{device_id}.pt")


                from joblib import Parallel, delayed
                batch_num = len(train_dataset) // batch_size
                gpu_num = 4
                Parallel(n_jobs=20)(delayed(sketching)(batch_idx, device_id, k=cfg.selection.k) for (batch_idx, device_id) in itertools.product(range(batch_num), range(gpu_num)))
                G_list = [torch.load(f"{leverage_score_model.output_dir}/sketched_grads_batch_idx={batch_idx}_{device_id}.pt", map_location="cpu") for (batch_idx, device_id) in itertools.product(range(31), range(4))]
                G = torch.cat(G_list, dim=0)
                d = cfg.selection.d
                torch.save(G, f"{leverage_score_model.output_dir}/G_d={d}.pt")
            #! testing part
            class_unconditioned_sampling(leverage_score_model, sample_wise_loss, train_dataset, eb_dataset_dict, output_dir, cfg, task, unpretrained_eb_dataset_dict)
            # if task == "classification":
                # oversample_class_conditioned(leverage_score_model, sample_wise_loss, train_dataset, eb_dataset_dict, output_dir, cfg, unpretrained_eb_dataset_dict)
                # class_conditioned_sampling(leverage_score_model, sample_wise_loss, train_dataset, eb_dataset_dict, output_dir, cfg, unpretrained_eb_dataset_dict)

    elif method == "leverage_score_change":
        eb_dataset_dict = load_eb_dataset_cfg(cfg, device="cuda")
        unpertrained_cfg = copy.deepcopy(cfg)
        del unpertrained_cfg.pretraining
        unpretrained_eb_dataset_dict = load_eb_dataset_cfg(unpertrained_cfg, device="cuda")
        logger.info("leverage_score_change")
        # sampling using leverage_score_pretrained - leverage_score_finetuned
        pretrained_cfg = copy.deepcopy(cfg)
        # change source to pretrained
        pretrained_cfg.selection.source = "pretrained"
        # delete the pretraining
        del pretrained_cfg.pretraining
        pretrained_output_dir = get_grad_dir(pretrained_cfg)
        assert cfg.selection.source == "finetuned"
        finetuned_output_dir = get_grad_dir(cfg)
        # check G
        logger.info(f"pretrained_output_dir: {pretrained_output_dir}")
        pretrained_G = torch.load(f"{pretrained_output_dir}/G.pt")
        finetuned_G = torch.load(f"{finetuned_output_dir}/G.pt")

        assert cfg.selection.use_residual == False
        diff_G = pretrained_G - finetuned_G
        diff_ls = torch.norm(diff_G, dim=1, p=2)
        prob = diff_ls / diff_ls.sum()

        assert not torch.isnan(prob).any()

        m = int(cfg.selection.fraction * len(train_dataset))
        idxes = np.random.choice(len(train_dataset), size=m, replace=False, p=prob.cpu().detach().numpy())
        test_eb(unpretrained_eb_dataset_dict, idxes, weights=None, use_weights=False, name="class_unconditioned", task=task)

    elif method == "random":
        logger.info("random")
        logger.info(f"task={task}")
        eb_dataset_dict = load_eb_dataset_cfg(cfg, device="cuda")
        is_pretrained = hasattr(cfg, "pretraining")
        if is_pretrained:
            unpertrained_cfg = copy.deepcopy(cfg)
            del unpertrained_cfg.pretraining
            unpretrained_eb_dataset_dict = load_eb_dataset_cfg(unpertrained_cfg, device="cuda")
        else:
            unpretrained_eb_dataset_dict = None
        class_conditioned_idxes = select_eb_random(eb_dataset_dict, m=cfg.selection.fraction, split="train", class_conditioned=True, seed=cfg.seed)
        class_unconditioned_idxes = select_eb_random(eb_dataset_dict, m=cfg.selection.fraction, split="train", class_conditioned=False, seed=cfg.seed)
        torch.save(class_conditioned_idxes, f"{output_dir}/c_conditioned_idxes.pt")
        torch.save(class_unconditioned_idxes, f"{output_dir}/c_unconditioned_idxes.pt")
        #TODO: clean the hard code
        test_as_val = True
        logger.info(f"test_as_val: {test_as_val}")
        for tune in [False]:
            from linear_probe import test_eb
            test_eb(eb_dataset_dict,
                    class_conditioned_idxes,
                    weights=None,
                    seed=cfg.seed,
                    use_weights=False,
                    name="class_conditioned",
                    task=task,
                    tune=tune,
                    dataset_name=cfg.dataset.name,
                    test_as_val=test_as_val,
                    )
            test_eb(eb_dataset_dict,
                    class_unconditioned_idxes,
                    weights=None,
                    seed=cfg.seed,
                    use_weights=False,
                    name="class_unconditioned",
                    task=task,
                    tune=tune,
                    dataset_name=cfg.dataset.name,
                    test_as_val=test_as_val,
                    )
        wandb.finish()

    elif method == "residual":
        eb_dataset_dict = load_eb_dataset_cfg(cfg, device="cuda")
        linear_probing_model = test_eb_full(eb_dataset_dict, split="train", task=task)
        train_X = eb_dataset_dict["train"]["X"].cpu().detach()
        train_Y = eb_dataset_dict["train"]["Y"].cpu().detach().long()
        sample_wise_logit = linear_probing_model.predict_proba(train_X)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        sample_wise_loss = loss_fn(torch.tensor(sample_wise_logit), torch.tensor(train_Y)).cpu().detach()

        m = int(cfg.selection.fraction * len(train_X))
        idxes = select_by_prob(sample_wise_loss, m=m, targets=train_Y, class_conditioned=False, selected_idxes=[])
        test_eb(eb_dataset_dict, idxes, weights=None, use_weights=False, name="class_unconditioned", task=task)
        idxes = select_by_prob(sample_wise_loss, m=m, targets=train_Y, class_conditioned=True, selected_idxes=[])
        test_eb(eb_dataset_dict, idxes, weights=None, use_weights=False, name="class_conditioned", task=task)

    elif method == "leverage_score" and cfg.selection.space == "feature":
        from methods.leverage_score import LeverageScoreFeature
        from methods.leverage_score import class_unconditioned_sampling, class_conditioned_sampling, oversample_class_conditioned
        eb_dataset_dict = load_eb_dataset_cfg(cfg, device="cuda")

        residual_preselection = hasattr(cfg.selection, "preselection") and cfg.selection.preselection.method == "residual"

        if cfg.selection.use_residual or residual_preselection:
            linear_probing_model = test_eb(eb_dataset_dict, np.arange(len(eb_dataset_dict["train"]["X"])), weights=None, use_weights=False, name="full", task=task, seed=cfg.seed, dataset_name=cfg.dataset.name)
            train_X = eb_dataset_dict["train"]["X"].cpu().detach().numpy()
            train_Y = eb_dataset_dict["train"]["Y"].cpu().detach().numpy().astype(int)
            sample_wise_loss = get_sample_wise_loss(linear_probing_model, train_X, train_Y, task=task)
        else:
            sample_wise_loss = None

        if hasattr(cfg.selection, "preselection"):
            logger.info(f"preselection: {cfg.selection.preselection}")
            logger.info(f"fraction: {cfg.selection.preselection.fraction}")
            assert cfg.selection.preselection.method in ["residual", "random", "submodular"]
            pre_fraction = cfg.selection.preselection.fraction
            if "x" in pre_fraction:
                pre_fraction = int(pre_fraction.replace("x", "")) * cfg.selection.fraction

            pre_method = cfg.selection.preselection.method
            assert pre_fraction >= cfg.selection.fraction, f"pre_fraction: {pre_fraction} should be larger than cfg.selection.fraction: {cfg.selection.fraction}"
            logger.info(f"pre_fraction: {pre_fraction}")

            pre_m = int(pre_fraction * len(eb_dataset_dict["train"]["X"]))
            if pre_method == "residual":
                preselected_idxes = sample_wise_loss.argsort()[:pre_m]
            elif pre_method == "random":
                preselected_idxes = np.random.choice(len(eb_dataset_dict["train"]["X"]), size=pre_m, replace=False)
            elif pre_method == "submodular":
                from apricot import FacilityLocationSelection
                # apricot.functions.maxCoverage.MaxCoverageSelection
                from apricot import MaxCoverageSelection
                train_X = eb_dataset_dict["train"]["X"]
                train_X_np = train_X.cpu().detach().numpy()
                preselector = MaxCoverageSelection(pre_m).fit(train_X_np)
                preselected_idxes = preselector.ranking
            else:
                raise NotImplementedError
            test_eb(eb_dataset_dict, preselected_idxes, weights=None, seed=cfg.seed, use_weights=False, name="preselection", task=task, dataset_name=cfg.dataset.name)
            preselected_dict = copy.deepcopy(eb_dataset_dict)
            preselected_dict["train"]["X"] = preselected_dict["train"]["X"][preselected_idxes]
            preselected_dict["train"]["Y"] = preselected_dict["train"]["Y"][preselected_idxes]


        # TODO: make this better
        # from hydra_zen import builds, instantiate
        # import inspect
        # ----------
        constructor_params = inspect.signature(LeverageScoreFeature.__init__).parameters
        selection_cfg = {k: v for k, v in cfg.selection.items() if k in constructor_params}
        LeverageScoreFeatureConfig = builds(
            LeverageScoreFeature,
            **selection_cfg,
            wandb_dir=wandb_dir,
            dataset_name=cfg.dataset.name,
            num_classes=cfg.dataset.num_classes,
            populate_full_signature=True  # This ensures all parameters are captured if needed
        )
        sel_method = instantiate(LeverageScoreFeatureConfig)
        # ----------
        if hasattr(cfg.selection, "preselection"):
            sel_method.setup(dataset_dict=preselected_dict, residual=sample_wise_loss)
        else:
            sel_method.setup(dataset_dict=eb_dataset_dict, residual=sample_wise_loss)
        sel_method.plot()
        sample_wise_loss = None
        unpretrained_eb_dataset_dict = None
        region_cfg = cfg.selection.preselection.region if hasattr(cfg.selection, "preselection") and hasattr(cfg.selection.preselection, "region") else None
        class_unconditioned_sampling(sel_method, sample_wise_loss, eb_dataset_dict, output_dir, cfg, task, unpretrained_eb_dataset_dict, preselected_idxes=preselected_idxes, region_cfg=region_cfg)
        # if task == "classification":
            # oversample_class_conditioned(sel_method, sample_wise_loss, eb_dataset_dict, output_dir, cfg, unpretrained_eb_dataset_dict)
            # class_conditioned_sampling(sel_method, sample_wise_loss, eb_dataset_dict, output_dir, cfg, unpretrained_eb_dataset_dict)

    elif method == "modified_leverage_score":
        wandb.init(project=cfg.wandb.project,
                    entity=cfg.wandb.entity,
                    name=cfg_hash)
        from methods.leverage_score import compute_leverage
        from methods.row_sampling import leverage_sampling, l2s_sampling, uniform_sampling
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        # def lev_neg_log_likelihood(theta, X, y, ind, w):
        #     z = np.dot(X, theta)
        #     log_sum_exp = np.logaddexp(0, z)
        #     return -np.sum(y * z)
            # return -np.sum(w*y[ind] * z[ind] - w*log_sum_exp[ind])

        # def logistic_regression_loss(theta, X, y, ind, w):
        #     z = np.dot(X, theta)
        #     z = sigmoid(z)
        #     return -np.sum(w * y[ind] * np.log(z[ind]) + w * (1 - y[ind]) * np.log(1 - z[ind]))
        def lev_neg_log_likelihood(theta, X, y, ind, w):
            z = np.dot(X, theta)
            log_sum_exp = np.logaddexp(0, z)
            return -np.sum(w*y[ind] * z[ind] - w*log_sum_exp[ind])

        def lev_gradient(theta, X, y, ind, w):
            z = np.dot(X, theta)
            return np.dot(X.T[:,ind], np.dot(np.diag(w),(sigmoid(z) - y)[ind]))

        def lev_neg_log_likelihood_torch(theta, X, y, ind, w):
            X = X.float()
            theta = theta.float()
            z = torch.matmul(X, theta)
            log_sum_exp = torch.logsumexp(torch.stack([torch.zeros_like(z), z]), dim=0)
            return -torch.sum(w * y[ind] * z[ind] - w * log_sum_exp[ind])

        # def lev_gradient(theta, X, y, ind, w):
        #     z = np.dot(X, theta)
        #     return np.dot(X.T[:,ind], np.dot(np.diag(w),(sigmoid(z) - y)[ind]))

        def eval_model(model, X, y, ind=None, split="test", task="classification"):
            logger.debug(X.shape)
            logger.debug(y.shape)
            if task == "classification":
                y_hat = model.predict(X)
                logger.debug(y_hat.shape)
                acc = accuracy_score(y, y_hat)
                wandb.log({f"class_unconditioned/use_weights=False/{split}/acc": acc})
            elif task == "regression":
                y_hat = model.predict(X)
                mse = mean_squared_error(y, y_hat)
                wandb.log({f"class_unconditioned/use_weights=False/{split}/mse": mse})

        eb_dataset_dict = load_eb_dataset_cfg(cfg, device="cuda")
        train_X = eb_dataset_dict["train"]["X"]
        train_Y = eb_dataset_dict["train"]["Y"]
        m = int(cfg.selection.fraction * len(train_X))
        if cfg.debug:
            train_X = train_X[:100]
            train_Y = train_Y[:100]
        train_X = make_numpy(train_X)
        train_Y = make_numpy(train_Y)
        logger.info("train_X.shape: %s", train_X.shape)
        logger.info("m, %d", m)
        if cfg.selection.use_residual:
            linear_probing_model = test_eb_full(eb_dataset_dict, split="train", task="classification")
            sample_wise_loss = get_sample_wise_loss(linear_probing_model, train_X, train_Y)
        else:
            sample_wise_loss = None

        ind, sel_w = leverage_sampling(data=train_X, size=m, sample_wise_loss=sample_wise_loss, use_residual=False)
        # train_X = make_numpy(train_X)
        # train_Y = make_numpy(train_Y)
        # sel_X = train_X[ind]
        # sel_Y = train_Y[ind]
        # logger.info("begin to minimize")
        initial_theta = np.zeros((train_X.shape[1]))
        # theta = lr_model.coef_.T
        result_lev = minimize(lev_neg_log_likelihood, initial_theta, args=(train_X, train_Y, ind, sel_w), options={'disp': True, 'maxiter': 1000}, method="BFGS")
        # jac is not needed
        theta = result_lev.x
        logger.info(theta.shape)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(theta, f"{output_dir}/theta.npy")
        lr_model = LogisticRegression()
        dummy_X = np.zeros((2, train_X.shape[1]))
        dummy_y = np.array([0, 1])
        lr_model.fit(dummy_X, dummy_y)
        lr_model.coef_ = theta.reshape(1, -1)
        lr_model.intercept_ = 0

        test_X = make_numpy(eb_dataset_dict["test"]["X"])
        test_Y = make_numpy(eb_dataset_dict["test"]["Y"])
        eval_model(lr_model, test_X, test_Y, split="test", task=task)

    elif method == "submodular":
        eb_dataset_dict = load_eb_dataset_cfg(cfg, device="cuda")
        train_X = eb_dataset_dict["train"]["X"]
        train_X_np = train_X.cpu().detach().numpy()
        train_Y = eb_dataset_dict["train"]["Y"]
        train_Y_np = train_Y.cpu().detach().numpy()
        if type(cfg.selection.fraction) == float:
            m = int(cfg.selection.fraction * len(train_X))
        else:
            m = cfg.selection.fraction
        logger.info(f"m: {m}")
        if cfg.selection.submodular_method == "facility_location":
            from apricot import FacilityLocationSelection
            selector = FacilityLocationSelection(m).fit(train_X_np)
            idxes = selector.ranking
            test_eb(eb_dataset_dict, idxes, weights=None, use_weights=False, name="class_unconditioned", task=task, seed=cfg.seed, dataset_name=cfg.dataset.name)
        elif cfg.selection.submodular_method == "max_coverage":
            from apricot import MaxCoverageSelection
            idxes = MaxCoverageSelection(m).fit(train_X_np, y=train_Y_np)
            test_eb(eb_dataset_dict, idxes, weights=None, use_weights=False, name="class_unconditioned", task=task, seed=cfg.seed, dataset_name=cfg.dataset.name)
    elif method == "cov_greedy":
        from methods.cov_greedy import cov_greedy_maxproj, cov_greedy_weighted_by_gaps
        from methods.cov_ga import cov_ga
        eb_dataset_dict = load_eb_dataset_cfg(cfg, device="cuda")
        if type(cfg.selection.fraction) == float:
            m = int(cfg.selection.fraction * len(eb_dataset_dict["train"]["X"]))
        else:
            m = cfg.selection.fraction
        m = int(cfg.selection.fraction * len(eb_dataset_dict["train"]["X"]))
        train_X = eb_dataset_dict["train"]["X"].cpu().detach().numpy()
        idxes = cov_greedy_weighted_by_gaps(train_X, m)
        # idxes = select_rows_based_on_eigenvectors()
        test_eb(eb_dataset_dict, idxes, weights=None, use_weights=False, name="class_unconditioned", task=task, seed=cfg.seed, dataset_name=cfg.dataset.name)
    elif method == "cov_ga":
        from methods.cov_ga import cov_ga
        eb_dataset_dict = load_eb_dataset_cfg(cfg, device="cuda")
        if type(cfg.selection.fraction) == float:
            m = int(cfg.selection.fraction * len(eb_dataset_dict["train"]["X"]))
        else:
            m = cfg.selection.fraction
        m = int(cfg.selection.fraction * len(eb_dataset_dict["train"]["X"]))
        train_X = eb_dataset_dict["train"]["X"].cpu().detach().numpy()
        idxes = cov_ga(train_X, m)
        test_eb(eb_dataset_dict, idxes, weights=None, use_weights=False, name="class_unconditioned", task=task, seed=cfg.seed, dataset_name=cfg.dataset.name)

    elif method in ["Uniform", "DeepFool", "Glister", "GraNd", "Forgetting"]:
        #! DeepCore
        """_summary_
            go the ./libs/DeepCore, run all.slurm
            python check.py
        """
        # libs/DeepCore/result/CIFAR10_ResNet18_ContextualDiversity_exp0_epoch0_2024-04-23 14:20:39.498353_0.1_3_0.000000.ckpt
        # match dataset model method _0_float.ckpt
        # {dst}_{net}_{mtd}_exp{exp}_epoch{epc}_{dat}_{fr}_{seed}_{acc}
        # match dst, net, mtf seed
        from methods.deepcore_methods import deepcore_load
        dataset = cfg.dataset.name
        backbone = cfg.backbone.name
        method = cfg.selection.method
        m = cfg.selection.fraction
        if type(m) == float:
            m = int(m * len(train_dataset))
        for test_as_val in [True]:
            indexes, weights = deepcore_load(dataset=dataset, backbone=backbone, method=method, m=m, seed=cfg.seed, test_as_val=test_as_val)
            eb_dataset_dict = load_eb_dataset_cfg(cfg, device="cuda")
            weights = torch.ones(len(indexes))
            from linear_probe import test_eb
            test_eb(eb_dataset_dict, indexes.copy(), weights=None, use_weights=False, name="class_unconditioned", task=task, seed=cfg.seed, dataset_name=cfg.dataset.name, test_as_val=test_as_val)
    else:
        raise ValueError(f"method {method} not supported")

    # torch.save(indexes, f"{output_dir}/idxes.pt")
    wandb.finish()
    return
    # debug = False
    # if debug:
    #     output_dir = f"{output_dir}/debug"
    #     os.makedirs(output_dir, exist_ok=True)
    #     X = X[:100]
    #     Cov = X.T @ X
    #     eigenvalues, eigenvectors = np.linalg.eig(Cov)
    #     eigenvectors = torch.from_numpy(eigenvectors)
    #     eigenvalues = torch.from_numpy(eigenvalues)

    # d, p = X.shape
    # #! CHECK
    # model = MyBounds(d, p, final_fraction, c, s_eps, eigenvectors, eigenvalues, X, s_threshold)
    # optimizer = torch.optim.Adam([
    #     {'params': model.s, 'lr': s_lr},
    #     {'params': model.gamma, 'lr': gamma_lr}
    # ], weight_decay=0)

    # eigenvalues = eigenvalues.cpu().detach().numpy()
    # eigenvectors = eigenvectors.cpu().detach().numpy()
    # # try minimize
    # loss_dict = defaultdict()
    # iter_count = 0
    # def callback(xk):
    #     # print loss_dict
    #     logger.info("loss: %f", loss_dict[iter_count])
    #     iter_count += 1
    #     return

    # X = X.float()
    # Y = Y.float()

    # n, p = X.shape
    # s0 = np.ones((n)) * 0.99 + np.random.rand((n)) * 0.01
    # # solve the original problem as theta0
    # theta0 = Ridge().fit(X.cpu().detach().numpy(), Y.cpu().detach().numpy()).coef_
    # gamma0 = np.ones((p))
    # # for s0 \in [0, 1]^n, theta0 \in R^p, gamma0 \in R^p
    # x0 = torch.cat([torch.from_numpy(s0).float(), torch.from_numpy(theta0).float(), torch.from_numpy(gamma0).float()])
    # X = X.cpu().detach().numpy()
    # Y = Y.cpu().detach().numpy()
    lib = "torch"
    if lib == "jax":
        def save_with_jit(x):
            host_callback.call(lambda x: np.save(f"{output_dir}/s.npy", x), x)

        def fun(x, data):
            X, Y, eigenvectors, eigenvalues, cov_ratio, l1_ratio, y_loss_ratio = data
            n, p = X.shape
            # s = x[:n]
            # theta = x[n:n+p]
            # gamma = x[n+p:]
            # s = jax.lax.slice(x, 0, n)
            # theta = jax.lax.slice(x, n, n+p)
            # gamma = jax.lax.slice(x, n+p, n+p+p)
            s = lax.dynamic_slice(x, (0,), (n,))
            theta = lax.dynamic_slice(x, (n,), (p,))
            gamma = lax.dynamic_slice(x, (n + p,), (x.shape[0] - n - p,))
            #
            S = jnp.diag(s)
            Cov_S = X.T @ S @ X
            Cov_S_eigvectors = Cov_S @ eigenvectors
            # quadratic_form = torch.einsum('ij,ij->j', eigenvectors, Cov_S_eigvectors)
            quadratic_form = jnp.einsum('ij,ij->j', eigenvectors, Cov_S_eigvectors)
            # cov_loss = torch.norm(quadratic_form - gamma * eigenvalues, p=2) / d
            cov_loss = jnp.linalg.norm(quadratic_form - gamma * eigenvalues, ord=2) / d
            # calulate y_loss
            y_loss = jnp.linalg.norm(S @ X @ theta - Y, ord=2) / n
            l1_loss = jnp.linalg.norm(s, ord=1) / n
            l0_ratio = jnp.linalg.norm(s, ord=0) / n
            loss = cov_loss * cov_ratio + y_loss * y_loss_ratio + l1_loss * l1_ratio
            jax.debug.print("loss: {}", loss)
            jax.debug.print("cov_loss: {}", cov_loss)
            jax.debug.print("y_loss: {}", y_loss)
            jax.debug.print("l1_loss: {}", l1_loss)
            jax.debug.print("l0_ratio: {}", l0_ratio)
            # save_with_jit(s)
            return loss
        n, p = X.shape
        print(X.shape, Y.shape, eigenvectors.shape, eigenvalues.shape)
        X = jnp.array(X)
        Y = jnp.array(Y)
        eigenvectors = jnp.array(eigenvectors)
        eigenvalues = jnp.array(eigenvalues)
        # bounds = [(0, 1)] * n + [(-np.inf, np.inf)] * p + [(1/c, c)] * p
        x0 = jnp.array(x0)


        def projection_bounds(x, hyperparams_proj):
            # n, p, c = hyperparams_proj
            # n = 49500
            # p = 2048
            # # lax
            # s = lax.dynamic_slice(x, (0,), (n,))
            # theta = lax.dynamic_slice(x, (n,), (p,))
            # gamma = lax.dynamic_slice(x, (n + p,), (n + p + p,))
            # p_s = jnp.clip(s, 0, 1)
            # p_theta = theta
            # p_gamma = jnp.clip(gamma, 1/c, c)
            # vector = jnp.concatenate([p_s, p_theta, p_gamma]).reshape(-1)
            return x
        pg = ProjectedGradient(fun, projection_bounds, maxiter=1000, verbose=1)
        cov_ratio = cfg.selection.cov_ratio
        l1_ratio = cfg.selection.l1_ratio
        y_loss_ratio = cfg.selection.y_loss_ratio
        pg_solution = pg.run(x0, hyperparams_proj=(n, p, c), data=(X, Y, eigenvectors, eigenvalues, cov_ratio, l1_ratio, y_loss_ratio))

        # pkl save
        with open(f"{output_dir}/pg_solution.pkl", 'wb') as file:
            pkl.dump(pg_solution, file)

    elif lib == "GeoTorch":
        from geotorch.optim.rsgd import RiemannianSGD
        print(x, x.shape)
        class LinearModel(nn.Module):
            def __init__(self, n, p, c, cov_ratio, l1_ratio, y_loss_ratio, eigenvectors, eigenvalues):
                super(LinearModel, self).__init__()
                self.n = n
                self.p = p
                self.c = c
                self.s = nn.Parameter(torch.ones((n)).float() * 0.99 + torch.rand((n)).float() * 0.01)
                self.theta = nn.Parameter(torch.zeros((p)).float())
                self.gamma = nn.Parameter(torch.ones((p)).float())

                self.cov_ratio = cov_ratio
                self.l1_ratio = l1_ratio
                self.y_loss_ratio = y_loss_ratio
                self.eigenvectors = eigenvectors
                self.eigenvalues = eigenvalues

            def forward(self, X, Y):
                eigenvectors = self.eigenvectors
                eigenvalues = self.eigenvalues
                cov_ratio = self.cov_ratio
                l1_ratio = self.l1_ratio
                y_loss_ratio = self.y_loss_ratio
                n, p = X.shape
                S = torch.diag(self.s)
                Cov_S = X.T @ S @ X
                Cov_S_eigvectors = Cov_S @ eigenvectors
                quadratic_form = torch.einsum('ij,ij->j', eigenvectors, Cov_S_eigvectors)
                cov_loss = torch.norm(quadratic_form - self.gamma * eigenvalues, p=2) / d
                # calulate y_loss
                y_loss = torch.norm(S @ X @ self.theta - Y, p=2) / n
                l1_loss = torch.norm(self.s, p=1) / n
                l0_ratio = torch.norm(self.s, p=0) / n
                loss = cov_loss * cov_ratio + y_loss * y_loss_ratio + l1_loss * l1_ratio
                return loss

    elif lib == "torch":
        best_score_dict = defaultdict(lambda: 1e32)
        for i in range(max_epochs):
            optimizer.zero_grad()
            loss = model(epoch=i)
            # add gaussian noise to the data

            loss.backward()
            optimizer.step()

            ratio = model.l0_ratio.item()
            cov_loss = model.cov_loss.item()
            # 0.0002
            if ratio > 0.01:
                floor_ratio = np.floor(ratio * 1e2) / 1e2
            elif ratio > 0.001:
                floor_ratio = np.floor(ratio * 1e3) / 1e3
            elif ratio > 0.0001:
                floor_ratio = np.floor(ratio * 1e4) / 1e4
            floor_sample = np.floor(floor_ratio * d).astype(int)
            logger.info("floor_ratio: %f", floor_ratio)
            if cov_loss < best_score_dict[floor_ratio] and False:
                # recalculate the cov_loss
                save_dir = f"{output_dir}/ratio={floor_ratio}/"
                os.makedirs(save_dir, exist_ok=True)
                s = model.s.cpu().detach().numpy()
                _, train_index = np.sort(s)[::-1], np.argsort(s)[::-1]
                train_index = train_index[:floor_sample]
                train_index = torch.from_numpy(train_index.copy()).long().cuda()

                # real_model = copy.deepcopy(model)
                with torch.no_grad():
                    real_s = torch.zeros_like(model.s)
                    # do the optimization for the real s
                    real_s[train_index] = 1
                    real_S = torch.diag(real_s).cuda()
                    real_SX = real_S @ X
                    real_Cov_S = real_SX.T @ real_SX
                    real_Cov_S_eigvectors = real_Cov_S @ eigenvectors
                    real_quadratic_form = torch.einsum('ij,ij->j', eigenvectors, real_Cov_S_eigvectors).cuda()
                    real_cov_loss = torch.norm(real_quadratic_form - model.gamma.cuda() * eigenvalues, p=2) / d


                if real_cov_loss < best_score_dict[floor_ratio]:
                    best_score_dict[floor_ratio] = real_cov_loss.item()
                    train_index_len = len(train_index)

                    assert train_index_len <= floor_ratio * d, f"{train_index_len} {floor_ratio} {d}"
                    torch.save(model.s, f"{save_dir}/s.pt")
                    torch.save(model.gamma, f"{save_dir}/gamma.pt")
                    torch.save(train_index, f"{save_dir}/train_index.pt")
                    torch.save(model, f"{save_dir}/model.pt")
                    logging.critical("best score: %f", best_score_dict[floor_ratio])
                    logging.critical("floor_ratio: %f", floor_ratio)

                    with open(f"{output_dir}/best_score_dict.yaml", 'w', encoding='utf-8') as file:
                        yaml.dump(best_score_dict, file)
                    plt.figure()
                    plt.plot(best_score_dict.keys(), best_score_dict.values())
                    plt.savefig(f"{output_dir}/best_score_dict.png")
                    wandb.log({"best_score_dict": wandb.Image(f"{output_dir}/best_score_dict.png")})

            with torch.no_grad():
                # constrain s to be within [0, 1]
                model.s.data = torch.clamp(model.s.data, s_eps, 1)
                model.gamma.data = torch.clamp(model.gamma.data, 1/c, c)
    wandb.finish()

if __name__ == '__main__':
    main()