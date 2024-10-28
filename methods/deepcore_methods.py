import os
import torch
import pytorch_lightning as pl
import pickle as pkl
import yaml
import re
import sys
import glob
from omegaconf import OmegaConf
import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger = logging.getLogger()

def deepcore_load(dataset="cifar10",
                  backbone="resnet18",
                  method="Uniform",
                  m=0.02,
                  seed=0,
                  test_as_val=False,
                  layers=None,
                  c_conditioned=False,
                  ):
    # hardcode here
    assert test_as_val == True
    if dataset == "StanfordCars":
        rename_dict = {
            500: 0.0614,
            1000: 0.1228,
            1500: 0.1842,
            2000: 0.2456,
            2500: 0.3070,
            3000: 0.3684,
            3500: 0.4298,
            4000: 0.4912,
        }
        fraction = rename_dict[m]
    elif dataset == "cifar10":
        rename_dict = {
            50: 0.001,
            100: 0.002,
            200: 0.004,
            500: 0.01,
            1000: 0.02,
            1500: 0.03,
            2000: 0.04,
            2500: 0.05,
            3000: 0.06,
            3500: 0.07,
            4000: 0.08,
        }
        fraction = rename_dict[m]
    else:
        raise NotImplementedError
    dataset_rename = {
        #TODO: debug the dataset name
        "cifar10": "cifar10",
        "cifar100": "CIFAR100",
        "StanfordCars": "StanfordCars",
    }
    dataset = dataset_rename[dataset]
    backbone_rename = {
        "resnet18": "ResNet18",
        "resnet50": "ResNet50",
        "resnet101": "ResNet101",
        "clip-vit-base-patch32": "CLIP",
    }
    if layers == -1:
        backbone = f"Linear{backbone_rename[backbone]}"
    elif layers == -2:
        backbone = f"TwoLayer{backbone_rename[backbone]}"
    elif layers == -3:
        backbone = f"ThreeLayer{backbone_rename[backbone]}"
    else:
        backbone = backbone_rename[backbone]
    if c_conditioned:
        balance = "True"
    else:
        balance = "False"
    # libs/DeepCore/StanfordCars_LinearCLIP-TwoLayerCLIP-ThreeLayerCLIP-LinearResNet18-TwoLayerResNet18_df.csv
    with open(f"libs/DeepCore/{balance}_t2path.pkl", "rb") as file:
        t2path = pkl.load(file)
    print(t2path)
    # save to {balance}_t2path.csv
    t = (dataset, backbone, method, str(fraction), str(seed))
    path = t2path[t]
    ckpt = torch.load(path, map_location="cpu")
    indexes = ckpt["subset"]['indices']
    #TODO: need to fix this
    weights = None 
    logger.info(f"indexes: {indexes}")
    return indexes, weights


#!DEPECATED
def deepcore_select(cfg, train_dataset):
    sys.path.append("LABROOTs/data_pruning")
    import libs.DeepCore.deepcore.methods as methods
    from opt import get_cfg_hash
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    output_dir = f"outputs/{get_cfg_hash(cfg_dict)}"
    selection_args = {}
    backbone_args = {
        "backbone_version": cfg.backbone.version,
        "backbone_name": cfg.backbone.name,
        "backbone_pretrain_path": getattr(cfg.backbone, "pretrain_path", None),
    }
    dataset_args = {
        "dataset_name": cfg.dataset.name,
    }
    fraction = cfg.selection.fraction
    if cfg.debug:
        from data_utils import MySubset
        train_dataset = MySubset(train_dataset, list(range(100)))
        fraction = 0.5
        epochs = 1
    else:
        if isinstance(fraction, int):
            fraction = fraction / len(train_dataset)
    rename_dict = {
        "grand": "GraNd",
        "forgetting": "Forgetting",
    }
    # args = {
    #     "device": "cuda",
    # }
    method_name = rename_dict[cfg.selection.method]
    # method = methods.__dict__[args.selection](dst_train, args, args.fraction, args.seed, **selection_args)
    selection_method = methods.__dict__[method_name](
                                dst_train=train_dataset,
                                args=cfg.selection,
                                fraction=fraction,
                                random_seed=cfg.selection.seed,
                                epochs=200,
                                **selection_args,
                                **dataset_args,
                                **backbone_args)
    global_seed = torch.initial_seed()
    pl.seed_everything(cfg.selection.seed)
    res = selection_method.select()
    indexes = res["indices"]
    weights = res["scores"]
    pl.seed_everything(global_seed)
    dir_dict = {
        "root": "cached_datasets",
        "backbone": cfg.backbone.str,
        "dataset": cfg.dataset.str,
        "selection": cfg.selection.str,
    }
    # dir = os.path.join(*[f"{k}#{v}" for k, v in dir_dict.items() if k != "root"])
    # dir = f"{dir}/fraction={cfg.selection.fraction}"
    # add the root
    # dir = os.path.join(dir_dict["root"], dir)
    with open(os.path.join(output_dir, "dir_dict.yaml"), "w") as f:
        yaml.dump(dir_dict, f)
    torch.save(indexes, os.path.join(output_dir, "train_index.pt"))
    logger.info(f"indexes: {indexes}")
    logger.info(f"weights: {weights}")
    return indexes, weights

if __name__ == "__main__":
    deepcore_load()