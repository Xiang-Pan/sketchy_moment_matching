import pandas as pd

import numpy as np
import wandb
import itertools
from rich import print
from joblib import Memory
import copy
from dataclasses import dataclass
from typing import Any, List, Optional, Dict, Union
from prefect.tasks import task_input_hash
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
memory = Memory(location="cache", verbose=1)

import hydra
import omegaconf
from omegaconf import OmegaConf
from itertools import product
from joblib import Parallel, delayed

from dataclasses import dataclass, field

@dataclass
class SharedConfig:
    dataset__name: Any
    backbone__name: Any
    seed: List[int]
    selection__fraction: List[int]
    finetuning__max_epochs: Optional[Any] = field(default=None, metadata={"omit_if_none": True})
    finetuning__optimizer__lr: Optional[Any] = field(default=None, metadata={"omit_if_none": True})
    finetuning__optimizer__weight_decay: Optional[Any] = field(default=None, metadata={"omit_if_none": True})
    finetuning__layers: Optional[Any] = field(default=None, metadata={"omit_if_none": True})
    finetuning__optimizer__feature_lr_decay: Optional[Any] = field(default=None, metadata={"omit_if_none": True})
    finetuning__optimizer__feature_weight_decay: Optional[Any] = field(default=None, metadata={"omit_if_none": True})

@dataclass
class RowConfig:
    selection__method: str
    selection__c: Optional[float] = field(default=None, metadata={"omit_if_none": True})
    selection__sparse_scale: Optional[float] = field(default=None, metadata={"omit_if_none": True})
    selection__s_init_method: Optional[str] = field(default=None, metadata={"omit_if_none": True})
    selection__c_conditioned: Optional[Any] = field(default=None, metadata={"omit_if_none": True})
    selection__sketching_dim: Optional[int] = field(default=None, metadata={"omit_if_none": True})
    selection__cls_pretrain_size: Optional[str] = field(default=None, metadata={"omit_if_none": True})

@dataclass
class RowSweepConfig:
    selection__method: List[str]
    selection__c: Optional[List[Any]] = field(default_factory=lambda: [None], metadata={"omit_if_none": True})
    selection__sparse_scale: Optional[List[Any]] = field(default_factory=lambda: [None], metadata={"omit_if_none": True})
    selection__s_init_method: Optional[List[Any]] = field(default_factory=lambda: [None], metadata={"omit_if_none": True})
    selection__c_conditioned: Optional[List[Any]] = field(default_factory=lambda: [None], metadata={"omit_if_none": True})
    selection__sketching_dim: Optional[List[Any]] = field(default_factory=lambda: [None], metadata={"omit_if_none": True})
    selection__cls_pretrain_size: Optional[List[Any]] = field(default_factory=lambda: [None], metadata={"omit_if_none": True})
    finetuning__optimizer__feature_weight_decay: Optional[List[Any]] = field(default_factory=lambda: [None], metadata={"omit_if_none": True})
    tags: Optional[Any] = field(default_factory=lambda: [None], metadata={"omit_if_none": True})
    
    # skmm_v2 
    selection__block_size: Optional[List[Any]] = field(default_factory=lambda: [None], metadata={"omit_if_none": True})
    selection__temperature: Optional[List[Any]] = field(default_factory=lambda: [None], metadata={"omit_if_none": True})
    

@dataclass
class SklearnSummaryKeys:
    c_true_acc: str
    c_sampled_acc: str
    c_false_acc: str
    c_true_f1: str
    c_sampled_f1: str
    c_false_f1: str

@dataclass
class FTSummaryKeys:
    test_acc: str
    test_f1: str


@dataclass
class TableConfig:
    shared: SharedConfig
    rows: List[RowConfig]
    summary_keys: SklearnSummaryKeys


@dataclass
class StanfordCarsClipSklearnConfig(TableConfig):
    name: str = "StanfordCarsClipSklearn"
    shared: Any = field(default_factory=lambda: SharedConfig(
        dataset__name="StanfordCars",
        backbone__name="clip-vit-base-patch32",
        seed=[0, 1, 2, 3, 4],
        selection__fraction=[500, 1000, 2000, 3000, 4000],
    ))
    rows: List[RowConfig] = field(default_factory=lambda: [
        RowConfig(
            selection__method="random"
        ),
        RowConfig(
            selection__method="cov",
            selection__c=0.8,
            selection__sparse_scale=1e-3,
            selection__s_init_method="random_m"
        )
    ])
    summary_keys: Any = field(default_factory=lambda: SklearnSummaryKeys(
        c_true_acc="class_conditioned/use_weights=False/test_as_val=True/tune=False/test/acc",
        c_sampled_acc="sampled_0/use_weights=False/test_as_val=True/tune=False/test/acc",
        c_false_acc="class_unconditioned/use_weights=False/test_as_val=True/tune=False/test/acc",
        c_true_f1="class_conditioned/use_weights=False/test_as_val=True/tune=False/test/f1",
        c_sampled_f1="sampled_0/use_weights=False/test_as_val=True/tune=False/test/f1",
        c_false_f1="class_unconditioned/use_weights=False/test_as_val=True/tune=False/test/f1"
    ))

@dataclass
class StanfordCarsClipFT1Config():
    name: str = "StanfordCarsClipFT1"
    shared: Any = field(default_factory=lambda: SharedConfig(
        dataset__name="StanfordCars",
        backbone__name="clip-vit-base-patch32",
        seed=[0, 1, 2, 3, 4],
        selection__fraction=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        finetuning__max_epochs=[50],
        finetuning__layers=[-1],
    ))
    rows: List[Any] = field(default_factory=lambda: [
        RowConfig(
            selection__method="Uniform"
        ),
        RowConfig(
            selection__method="ContextualDiversity"
        ),
        RowConfig(
            selection__method="DeepFool"
        ),
        RowConfig(
            selection__method="Forgetting"
        ),
        RowConfig(
            selection__method="Herding"
        ),
        RowConfig(
            selection__method="GraNd"
        ),
        RowConfig(
            selection__method="Glister"
        ),
        RowConfig(
            selection__method="Uncertainty-Entropy"
        ),
        RowConfig(
            selection__method="Uncertainty-Margin"
        ),
        RowConfig(
            selection__method="Uncertainty-LeastConfidence"
        ),
        RowSweepConfig(
            selection__method=["SkMMv2"],
            selection__sketching_dim=[64, 128, 256, 512],
            selection__block_size=[5, 10, 15, 20],
            selection__temperature=[1e-3]
        ),
        RowSweepConfig(
            selection__method=["cov"],
            selection__c=[0.6, 0.7, 0.8, 0.9],
            selection__sparse_scale=[1e-3],
            selection__s_init_method=["random_m"],
            selection__sketching_dim=[16, 32, 64, 128, 256, 512],
            selection__c_conditioned=["sampled", False],
        )
    ])
    summary_keys: Any = field(default_factory=lambda: FTSummaryKeys(
        test_acc="test/acc",
        test_f1="test/f1"
    ))


@dataclass
class StanfordCarsResNet18FT1Config():
    name: str = "StanfordCarsResNet18FT1"
    shared: Any = field(default_factory=lambda: SharedConfig(
        dataset__name="StanfordCars",
        backbone__name="resnet18",
        seed=[0, 1, 2, 3, 4],
        selection__fraction=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        finetuning__max_epochs=[50],
        finetuning__layers=[-1],
        finetuning__optimizer__lr=[1e-2],
    ))
    rows: List[Any] = field(default_factory=lambda: [
        RowConfig(
            selection__method="Uniform"
        ),
        RowConfig(
            selection__method="ContextualDiversity"
        ),
        RowConfig(
            selection__method="Forgetting"
        ),
        RowConfig(
            selection__method="Herding"
        ),
        RowConfig(
            selection__method="GraNd"
        ),
        RowConfig(
            selection__method="Glister"
        ),
        RowConfig(
            selection__method="Uncertainty-Entropy"
        ),
        RowConfig(
            selection__method="Uncertainty-Margin"
        ),
        RowConfig(
            selection__method="Uncertainty-LeastConfidence"
        ),
        RowSweepConfig(
            selection__method=["cov"],
            selection__c=[0.6, 0.7, 0.8, 0.9],
            selection__sparse_scale=[1e-3],
            selection__s_init_method=["random_m"],
            selection__sketching_dim=[16, 32, 64, 128, 256, 512],
            selection__c_conditioned=["sampled"],
        ),
    ])
    summary_keys: Any = field(default_factory=lambda: FTSummaryKeys(
        test_acc="test/acc",
        test_f1="test/f1"
    ))



@dataclass
class CIFAR10ResNet18FT1Config():
    name: str = "CIFAR10ResNet18FT1"
    shared: Any = field(default_factory=lambda: SharedConfig(
        dataset__name="cifar10",
        backbone__name="resnet18",
        seed=[0, 1, 2, 3, 4],
        selection__fraction=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        finetuning__max_epochs=[200],
        finetuning__layers=[-1],
        finetuning__optimizer__lr=[1e-2],
    ))
    rows: List[Any] = field(default_factory=lambda: [
        RowConfig(
            selection__method="Uniform"
        ),
        RowConfig(
            selection__method="ContextualDiversity"
        ),
        RowConfig(
            selection__method="DeepFool"
        ),
        RowConfig(
            selection__method="Forgetting"
        ),
        RowConfig(
            selection__method="Herding"
        ),
        RowConfig(
            selection__method="GraNd"
        ),
        RowConfig(
            selection__method="Glister"
        ),
        RowConfig(
            selection__method="Uncertainty-Entropy"
        ),
        RowConfig(
            selection__method="Uncertainty-Margin"
        ),
        RowConfig(
            selection__method="Uncertainty-LeastConfidence"
        ),
        RowSweepConfig(
            selection__method=["cov"],
            selection__c=[0.6, 0.7, 0.8, 0.9],
            selection__sparse_scale=[1e-3],
            selection__s_init_method=["random_m"],
            selection__sketching_dim=[16, 32, 64, 128, 256, 512],
            selection__c_conditioned=["sampled"],
        )
    ])
    summary_keys: Any = field(default_factory=lambda: FTSummaryKeys(
        test_acc="test/acc",
        test_f1="test/f1"
    ))



@dataclass
class StanfordCarsClipFT2Config():
    name: str = "StanfordCarsClipFT2"
    shared: Any = field(default_factory=lambda: SharedConfig(
        dataset__name="StanfordCars",
        backbone__name="clip-vit-base-patch32",
        seed=[0, 1, 2, 3, 4],
        selection__fraction=[500, 1000, 2000, 3000, 4000],
        finetuning__max_epochs=[50],
        finetuning__layers=[-2],
        finetuning__optimizer__feature_lr_decay=1,
        finetuning__optimizer__lr=1e-2,
        finetuning__optimizer__feature_weight_decay=0,
    ))
    rows: List[Any] = field(default_factory=lambda: [
        RowConfig(
            selection__method="Uniform"
        ),
        RowConfig(
            selection__method="ContextualDiversity"
        ),
        RowConfig(
            selection__method="DeepFool"
        ),
        RowConfig(
            selection__method="Forgetting"
        ),
        RowConfig(
            selection__method="Herding"
        ),
        RowConfig(
            selection__method="GraNd"
        ),
        RowConfig(
            selection__method="Glister"
        ),
        RowConfig(
            selection__method="Uncertainty-Entropy"
        ),
        RowConfig(
            selection__method="Uncertainty-Margin"
        ),
        RowConfig(
            selection__method="Uncertainty-LeastConfidence"
        ),
        RowSweepConfig(
            selection__method=["SkMMv2"],
            selection__sketching_dim=[64, 128, 256, 512],
        ),
        # RowSweepConfig(
        #     selection__method=["cov"],
        #     selection__c=[0.6, 0.8, 0.9],
        #     selection__sparse_scale=[1e-3],
        #     selection__s_init_method=["random_m"],
        #     selection__sketching_dim=[64, 128, 256, 512],
        #     selection__c_conditioned=["sampled"],
        #     selection__cls_pretrain_size=["None"]
        # ),
        # RowSweepConfig(
        #     selection__method=["cov_ntk"],
        #     selection__c=[0.6, 0.7, 0.8, 0.9],
        #     selection__sparse_scale=[1e-3],
        #     selection__s_init_method=["random_m"],
        #     selection__sketching_dim=[64, 128, 256, 512],
        #     selection__c_conditioned=["sampled"],
        #     selection__cls_pretrain_size=["full"]
        # ),
        RowSweepConfig(
            selection__method=["cov_ntk"],
            selection__c=[0.6, 0.7, 0.8, 0.9],
            selection__sparse_scale=[1e-3],
            selection__s_init_method=["random_m"],
            selection__sketching_dim=[64, 128, 256, 512],
            selection__c_conditioned=["sampled"],
        ),
    ])
    summary_keys: Any = field(default_factory=lambda: FTSummaryKeys(
        test_acc="test/acc",
        test_f1="test/f1"
    ))

@dataclass
class StanfordCarsResNet18FT2SKMMv2Config():
    name: str = "StanfordCarsResNet18FT2SKMMv2"
    shared: Any = field(default_factory=lambda: SharedConfig(
        dataset__name="StanfordCars",
        backbone__name="resnet18",
        seed=[0, 1, 2, 3, 4],
        selection__fraction=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        finetuning__max_epochs=[50],
        finetuning__layers=[-2],
        finetuning__optimizer__lr=[1e-2],
        finetuning__optimizer__feature_lr_decay=1,
        finetuning__optimizer__feature_weight_decay=0,
    ))
    rows: List[Any] = field(default_factory=lambda: [
        RowSweepConfig(
            selection__method=["SkMMv2"],
            selection__sketching_dim=[32, 64, 128, 256, 512],
            selection__block_size=[5, 10, 15, 20],
            selection__temperature=[1e-3]
        ),
    ])
    summary_keys: Any = field(default_factory=lambda: FTSummaryKeys(
        test_acc="test/acc",
        test_f1="test/f1"
    ))



@dataclass
class StanfordCarsResNet18FT2Config():
    name: str = "StanfordCarsResNet18FT2"
    shared: Any = field(default_factory=lambda: SharedConfig(
        dataset__name="StanfordCars",
        backbone__name="resnet18",
        seed=[0, 1, 2, 3, 4],
        selection__fraction=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        finetuning__max_epochs=[50],
        finetuning__layers=[-2],
        finetuning__optimizer__lr=[1e-2],
        finetuning__optimizer__feature_lr_decay=1,
        finetuning__optimizer__feature_weight_decay=0,
    ))
    rows: List[Any] = field(default_factory=lambda: [
        RowConfig(
            selection__method="Uniform"
        ),
        RowConfig(
            selection__method="ContextualDiversity"
        ),
        RowConfig(
            selection__method="DeepFool"
        ),
        RowConfig(
            selection__method="Forgetting"
        ),
        RowConfig(
            selection__method="Herding"
        ),
        RowConfig(
            selection__method="GraNd"
        ),
        RowConfig(
            selection__method="Glister"
        ),
        RowConfig(
            selection__method="Uncertainty-Entropy"
        ),
        RowConfig(
            selection__method="Uncertainty-Margin"
        ),
        RowConfig(
            selection__method="Uncertainty-LeastConfidence"
        ),
        RowSweepConfig(
            selection__method=["SkMMv2"],
            selection__sketching_dim=[64, 128, 256, 512],
            selection__block_size=[5, 10, 15, 20],
            selection__temperature=[1e-3]
        ),
        RowSweepConfig(
            selection__method=["cov"],
            selection__c=[0.6, 0.8, 0.9],
            selection__sparse_scale=[1e-3],
            selection__s_init_method=["random_m"],
            selection__sketching_dim=[64, 128, 256, 512],
            selection__c_conditioned=["sampled"],
        ),
        RowSweepConfig(
            selection__method=["cov_ntk"],
            selection__c=[0.6, 0.7, 0.8, 0.9],
            selection__sparse_scale=[1e-3],
            selection__s_init_method=["random_m"],
            selection__sketching_dim=[64, 128, 256, 512],
            selection__c_conditioned=["sampled"],
        ),

    ])
    summary_keys: Any = field(default_factory=lambda: FTSummaryKeys(
        test_acc="test/acc",
        test_f1="test/f1"
    ))


@dataclass
class CIFAR10ResNet18FT2Config():
    name: str = "CIFAR10ResNet18FT2"
    shared: Any = field(default_factory=lambda: SharedConfig(
        dataset__name="cifar10",
        backbone__name="resnet18",
        seed=[0, 1, 2, 3, 4],
        selection__fraction=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        finetuning__max_epochs=[50],
        finetuning__layers=[-2],
        finetuning__optimizer__lr=[1e-2],
        finetuning__optimizer__feature_lr_decay=1,
        finetuning__optimizer__feature_weight_decay=0,
    ))
    rows: List[Any] = field(default_factory=lambda: [
        RowSweepConfig(
            selection__method=["Uniform"],
            selection__c_conditioned=[False, True],
        ),
        RowSweepConfig(
            selection__method=["ContextualDiversity"],
            selection__c_conditioned=[False, True],
        ),
        RowSweepConfig(
            selection__method=["DeepFool"],
            selection__c_conditioned=[False, True],
        ),
        RowSweepConfig(
            selection__method=["Forgetting"],
            selection__c_conditioned=[False, True],
        ),
        RowSweepConfig(
            selection__method=["Herding"],
            selection__c_conditioned=[False, True],
        ),
        RowSweepConfig(
            selection__method=["GraNd"],
            selection__c_conditioned=[False, True],
        ),
        RowSweepConfig(
            selection__method=["Glister"],
            selection__c_conditioned=[False, True],
        ),
        RowSweepConfig(
            selection__method=["Uncertainty-Entropy"],
            selection__c_conditioned=[False, True],
        ),
        RowSweepConfig(
            selection__method=["Uncertainty-Margin"],
            selection__c_conditioned=[False, True],
        ),
        RowSweepConfig(
            selection__method=["Uncertainty-LeastConfidence"],
            selection__c_conditioned=[False, True],
        ),
        RowSweepConfig(
            selection__method=["cov_ntk"],
            selection__c=[0.6, 0.7, 0.8, 0.9, 0.99],
            selection__sparse_scale=[1e-3],
            selection__s_init_method=["random_m"],
            selection__sketching_dim=[64, 128, 256, 512],
            selection__c_conditioned=["sampled", True],
        ),
        RowSweepConfig(
            selection__method=["cov_ntk_perclass"],
            selection__c=[0.6, 0.7, 0.8, 0.9, 0.99],
            selection__sparse_scale=[1e-3],
            selection__s_init_method=["random_m"],
            selection__sketching_dim=[32, 64, 128, 256, 512],
            selection__c_conditioned=[False],
        ),
        RowConfig(
            selection__method="random",
        ),
    ])
    summary_keys: Any = field(default_factory=lambda: FTSummaryKeys(
        test_acc="test/acc",
        test_f1="test/f1"
    ))

@dataclass
class CIFAR10ClipFT1Config():
    name: str = "CIFAR10ClipFT1"
    shared: Any = field(default_factory=lambda: SharedConfig(
        dataset__name="cifar10",
        backbone__name="clip-vit-base-patch32",
        seed=[0, 1, 2, 3, 4],
        selection__fraction=[50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        finetuning__max_epochs=[50],
        finetuning__layers=[-1],
    ))
    rows: List[Any] = field(default_factory=lambda: [
        RowConfig(
            selection__method="Uniform"
        ),
        RowConfig(
            selection__method="ContextualDiversity"
        ),
        RowConfig(
            selection__method="DeepFool"
        ),
        RowConfig(
            selection__method="Forgetting"
        ),
        RowConfig(
            selection__method="Herding"
        ),
        RowConfig(
            selection__method="GraNd"
        ),
        RowConfig(
            selection__method="Glister"
        ),
        RowConfig(
            selection__method="Uncertainty-Entropy"
        ),
        RowConfig(
            selection__method="Uncertainty-Margin"
        ),
        RowConfig(
            selection__method="Uncertainty-LeastConfidence"
        ),
        RowSweepConfig(
            selection__method=["SkMMv2"],
            selection__sketching_dim=[64, 128, 256, 512],
            selection__block_size=[5, 10, 15, 20],
            selection__temperature=[1e-3]
        )
    ])
    summary_keys: Any = field(default_factory=lambda: FTSummaryKeys(
        test_acc="test/acc",
        test_f1="test/f1"
    ))
        



@dataclass
class CIFAR10ResNet18FT2SKMMv2Config():
    name: str = "CIFAR10ResNet18FT2SKMMv2"
    shared: Any = field(default_factory=lambda: SharedConfig(
        dataset__name="cifar10",
        backbone__name="resnet18",
        seed=[0, 1, 2, 3, 4],
        selection__fraction=[50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        finetuning__max_epochs=[50],
        finetuning__layers=[-2],
        finetuning__optimizer__lr=[1e-2],
        finetuning__optimizer__feature_lr_decay=1,
        finetuning__optimizer__feature_weight_decay=0,
    ))
    rows: List[Any] = field(default_factory=lambda: [
        RowSweepConfig(
            selection__method=["Uniform"],
            selection__c_conditioned=[False],
        ),
        RowSweepConfig(
            selection__method=["ContextualDiversity"],
            selection__c_conditioned=[False],
        ),
        RowSweepConfig(
            selection__method=["DeepFool"],
            selection__c_conditioned=[False],
        ),
        RowSweepConfig(
            selection__method=["Forgetting"],
            selection__c_conditioned=[False],
        ),
        RowSweepConfig(
            selection__method=["Herding"],
            selection__c_conditioned=[False],
        ),
        RowSweepConfig(
            selection__method=["GraNd"],
            selection__c_conditioned=[False],
        ),
        RowSweepConfig(
            selection__method=["Glister"],
            selection__c_conditioned=[False],
        ),
        RowSweepConfig(
            selection__method=["Uncertainty-Entropy"],
            selection__c_conditioned=[False],
        ),
        RowSweepConfig(
            selection__method=["Uncertainty-Margin"],
            selection__c_conditioned=[False],
        ),
        RowSweepConfig(
            selection__method=["Uncertainty-LeastConfidence"],
            selection__c_conditioned=[False],
        ),
        RowSweepConfig(
            selection__method=["SkMMv2"],
            selection__sketching_dim=[64, 128, 256, 512],
            selection__block_size=[5, 10, 15, 20],
            selection__temperature=[1e-3]
        ),
        
    ])
    summary_keys: Any = field(default_factory=lambda: FTSummaryKeys(
        test_acc="test/acc",
        test_f1="test/f1"
    ))

@dataclass
class CIFAR10ResNet18FT2WDConfig():
    name: str = "CIFAR10ResNet18FT2WD"
    shared: Any = field(default_factory=lambda: SharedConfig(
        dataset__name="cifar10",
        backbone__name="resnet18",
        seed=[0, 1, 2, 3, 4],
        selection__fraction=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        finetuning__max_epochs=[50],
        finetuning__layers=[-2],
        finetuning__optimizer__lr=[1e-2],
        finetuning__optimizer__feature_lr_decay=1,
    ))
    rows: List[Any] = field(default_factory=lambda: [
        RowSweepConfig(
            selection__method=["random"],
            finetuning__optimizer__feature_weight_decay=[0, 1e-5, 1e-4, 1e-3],
        ),
        RowSweepConfig(
            selection__method=["cov_ntk"],
            selection__c=[0.6, 0.7, 0.8, 0.9, 0.99],
            selection__sparse_scale=[1e-3],
            selection__s_init_method=["random_m"],
            selection__sketching_dim=[64, 128, 256, 512],
            selection__c_conditioned=["sampled"],
            finetuning__optimizer__feature_weight_decay=[0, 1e-5, 1e-4, 1e-3],
        ),
    ])
    summary_keys: Any = field(default_factory=lambda: FTSummaryKeys(
        test_acc="test/acc",
        test_f1="test/f1"
    ))

@dataclass
class StanfordCarsResNet18FT2BaselinesConfig():
    name: str = "StanfordCarsResNet18FT2Baselines"
    shared: Any = field(default_factory=lambda: SharedConfig(
        dataset__name="StanfordCars",
        backbone__name="resnet18",
        seed=[0, 1, 2, 3, 4],
        selection__fraction=[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        finetuning__max_epochs=[50],
        finetuning__layers=[-2],
        finetuning__optimizer__lr=[1e-2],
        finetuning__optimizer__feature_lr_decay=1,
        finetuning__optimizer__feature_weight_decay=0,
    ))
    rows: List[Any] = field(default_factory=lambda: [
        RowConfig(
            selection__method="Uniform"
        ),
        RowConfig(
            selection__method="ContextualDiversity"
        ),
        RowConfig(
            selection__method="DeepFool"
        ),
        RowConfig(
            selection__method="Forgetting"
        ),
        RowConfig(
            selection__method="Herding"
        ),
        RowConfig(
            selection__method="GraNd"
        ),
        RowConfig(
            selection__method="Glister"
        ),
        RowConfig(
            selection__method="Uncertainty-Entropy"
        ),
        RowConfig(
            selection__method="Uncertainty-Margin"
        ),
        RowConfig(
            selection__method="Uncertainty-LeastConfidence"
        ),
    ])
    summary_keys: Any = field(default_factory=lambda: FTSummaryKeys(
        test_acc="test/acc",
        test_f1="test/f1"
    ))



@dataclass
class StanfordCarsClipFT3Config():
    name: str = "StanfordCarsClipFT3"
    shared: Any = field(default_factory=lambda: SharedConfig(
        dataset__name="StanfordCars",
        backbone__name="clip-vit-base-patch32",
        seed=[0, 1, 2, 3, 4],
        selection__fraction=[500, 1000, 2000, 3000, 4000],
        finetuning__max_epochs=[50],
        finetuning__layers=[-3],
        finetuning__optimizer__feature_lr_decay=1e-3,
    ))
    rows: List[Any] = field(default_factory=lambda: [
        RowConfig(
            selection__method="Uniform"
        ),
        RowConfig(
            selection__method="ContextualDiversity"
        ),
        RowConfig(
            selection__method="Forgetting"
        ),
        RowConfig(
            selection__method="Herding"
        ),
        RowConfig(
            selection__method="GraNd"
        ),
        RowConfig(
            selection__method="Glister"
        ),
        RowConfig(
            selection__method="Uncertainty-Entropy"
        ),
        RowConfig(
            selection__method="Uncertainty-Margin"
        ),
        RowConfig(
            selection__method="Uncertainty-LeastConfidence"
        ),
        RowSweepConfig(
            selection__method=["cov"],
            selection__c=[0.6, 0.8, 0.9],
            selection__sparse_scale=[1e-3],
            selection__s_init_method=["random_m"],
            selection__sketching_dim=[512],
            selection__c_conditioned=[False]
        ),
        RowConfig(
            selection__method="cov_ntk",
            selection__c=0.9,
            selection__sparse_scale=1e-3,
            selection__s_init_method="random_m",
            selection__c_conditioned="sampled",
            selection__sketching_dim=512,
        ),
        RowConfig(
            selection__method="cov_ntk",
            selection__c=0.8,
            selection__sparse_scale=1e-3,
            selection__s_init_method="random_m",
            selection__c_conditioned="sampled",
            selection__sketching_dim=512,
        ),
        RowConfig(
            selection__method="cov_ntk",
            selection__c=0.8,
            selection__sparse_scale=1e-3,
            selection__s_init_method="random_m",
            selection__c_conditioned="sampled",
            selection__sketching_dim=256,
        ),
        RowConfig(
            selection__method="cov_ntk",
            selection__c=0.8,
            selection__sparse_scale=1e-3,
            selection__s_init_method="random_m",
            selection__c_conditioned="sampled",
            selection__sketching_dim=64,
        ),
    ])
    summary_keys: Any = field(default_factory=lambda: FTSummaryKeys(
        test_acc="test/acc",
        test_f1="test/f1"
    ))


@dataclass
class StanfordCarsClipFT3WDConfig():
    name: str = "StanfordCarsClipFT3WD"
    shared: Any = field(default_factory=lambda: SharedConfig(
        dataset__name="StanfordCars",
        backbone__name="clip-vit-base-patch32",
        seed=[0, 1, 2, 3, 4],
        selection__fraction=[500, 1000, 2000, 3000, 4000],
        finetuning__max_epochs=[50],
        finetuning__layers=[-3],
        finetuning__optimizer__feature_lr_decay=1e-3,
        finetuning__optimizer__feature_weight_decay=1e-3,
    ))
    rows: List[RowConfig] = field(default_factory=lambda: [
        # ["Uniform", "ContextualDiversity", "GraNd", "Herding", "Forgetting", "Uncertainty-Entropy", "Uncertainty-Margin", "Uncertainty-LeastConfidence"]:
        RowConfig(
            selection__method="random",
        ),
        RowConfig(
            selection__method="cov_ntk",
            selection__c=0.9,
            selection__sparse_scale=1e-3,
            selection__s_init_method="random_m",
            selection__c_conditioned="sampled",
            selection__sketching_dim=512,
        ),
    ])
    summary_keys: Any = field(default_factory=lambda: FTSummaryKeys(
        test_acc="test/acc",
        test_f1="test/f1"
    ))



# Function to generate all combinations from a dictionary of lists
def generate_combinations(dict_list):
    keys, values = zip(*dict_list.items())
    values = [[v] if not isinstance(v, list) else v for v in values]
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combinations



def get_run_calls(table):
    # Generate run calls using itertools.product
    run_calls = []

    # Extract shared values and possible combinations
    shared_values = table["shared"]
    shared_combinations = generate_combinations(shared_values)

    # Extract row values and possible combinations
    rows = table["rows"]
    # No need to generate combinations if rows contain dictionaries without lists
    row_combinations = rows

    # Generate all combinations of shared values and row values
    for shared_comb, row in itertools.product(shared_combinations, row_combinations):
        logger.info(f"Shared: {shared_comb}, Row: {row}")
        # Combine shared_comb with non-list items in shared_values
        if type(row) == RowSweepConfig:
            logger.info(f"Row is RowSweepConfig")
        combined_shared = {**shared_values, **shared_comb}
        run_call = {
            "shared": combined_shared,
            "row": dict(row),
            "summary_keys": table["summary_keys"]
        }
        run_calls.append(run_call)
    return run_calls

#
# from prefect import task, flow
# @task(retries=3, cache_key_fn=task_input_hash)


class WarningFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.has_warning = False

    def filter(self, record):
        if record.levelno == logging.WARNING:
            self.has_warning = True
        return True

    def __enter__(self):
        logger.addFilter(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logger.removeFilter(self)

def conditional_cache(func):
    cached_func = memory.cache(func)

    def wrapper(*args, **kwargs):
        with WarningFilter() as warning_filter:
            result = func(*args, **kwargs)
            if not warning_filter.has_warning:
                return cached_func(*args, **kwargs)
            else:
                return result

    return wrapper

# @memory.cache
def fetch_run_call(run_call, stage):
    filters = run_call["shared"] | run_call["row"]
    filters = {k: v if v != "None" else None for k, v in filters.items()}
    graphql_filters = {f"config.{k}": {"$eq": v} for k, v in filters.items()}
    graphql_filters = {k.replace("__", "."): v for k, v in graphql_filters.items()}
    # if config.tags in graphql_filters, change to tags
    # if "config.tags" in graphql_filters:
        # graphql_filters["tags"] = graphql_filters.pop("config.tags")

    if runs := wandb.Api().runs(
        f"WANDB_ENTITY/data_pruning-{stage}", filters=graphql_filters,
    ):
        # do the manual filtering
        # make sure the k in the manual_filters not in the runs
        run = runs[0]
        summary = {
            f"{k}": run.summary.get(v, np.nan) for k, v in run_call["summary_keys"].items()
        }
        res = filters | summary
    else:
        logger.warning(f"Run not found for {filters}")
        res = filters | {k: np.nan for k in run_call["summary_keys"].keys()}
    res = pd.DataFrame([res])
    return res

def expand_configurations(config_list):
    expanded_configs = []

    for config in config_list:
        # Extract keys and their corresponding list values
        keys, value_lists = zip(*config.items())
        print(keys, value_lists)
        # Generate all possible combinations
        for combination in product(*value_lists):
            expanded_config = dict(zip(keys, combination))
            expanded_configs.append(expanded_config)

    return expanded_configs


from hydra.core.config_store import ConfigStore

@dataclass
class Config:
    table: Any
    stage: str = "finetuning"


# store the table config
cs = ConfigStore.instance()
cs.store(name="base", node=Config)
cs.store(group="table", name="StanfordCarsClipSklearn", node=StanfordCarsClipSklearnConfig)
cs.store(group="table", name="StanfordCarsClipFT1", node=StanfordCarsClipFT1Config)
cs.store(group="table", name="StanfordCarsClipFT2", node=StanfordCarsClipFT2Config)
cs.store(group="table", name="StanfordCarsClipFT3", node=StanfordCarsClipFT3Config)
cs.store(group="table", name="StanfordCarsClipFT3WD", node=StanfordCarsClipFT3WDConfig)
cs.store(group="table", name="StanfordCarsResNet18FT1", node=StanfordCarsResNet18FT1Config)
cs.store(group="table", name="StanfordCarsResNet18FT2", node=StanfordCarsResNet18FT2Config)
cs.store(group="table", name="StanfordCarsResNet18FT2Baselines", node=StanfordCarsResNet18FT2BaselinesConfig)
cs.store(group="table", name="StanfordCarsResNet18FT2SKMMv2", node=StanfordCarsResNet18FT2SKMMv2Config)

cs.store(group="table", name="CIFAR10ClipFT1", node=CIFAR10ClipFT1Config)
cs.store(group="table", name="CIFAR10ResNet18FT1", node=CIFAR10ResNet18FT1Config)
cs.store(group="table", name="CIFAR10ResNet18FT2", node=CIFAR10ResNet18FT2Config)
cs.store(group="table", name="CIFAR10ResNet18FT2WD", node=CIFAR10ResNet18FT2WDConfig)
cs.store(group="table", name="CIFAR10ResNet18FT2SKMMv2", node=CIFAR10ResNet18FT2SKMMv2Config)



def call(run_calls, stage):
    # Use joblib to parallelize the fetching of run calls
    df_list = Parallel(n_jobs=-1)(delayed(fetch_run_call)(run_call, stage) for run_call in run_calls)
    ori_df = pd.concat(df_list, ignore_index=True)
    return ori_df

def extract_mean(v):
    v = v.replace('\\textbf{', '').replace('\\underline{', '').replace('}', '')
    return v.split('±')[0]

def extract_var(v):
    v = v.replace('\\textbf{', '').replace('\\underline{', '').replace('}', '')
    return v.split('±')[1]

def highlight_max(df):
    for fraction in df.columns.values:
        # highlight the max acc
        frac_df = df[fraction].reset_index()
        non_cov_df = frac_df[~frac_df['selection'].str.contains("cov")]
        # get all test_acc
        frac_df = frac_df[frac_df['metric'] == 'test_acc']
        frac_df['mean'] = frac_df[fraction].apply(lambda x: extract_mean(x)).astype(float)
        frac_df['stderr'] = frac_df[fraction].apply(lambda x: extract_var(x)).astype(float)
        # get the largest mean
        frac_df = frac_df[frac_df['mean'] == frac_df['mean'].max()]
        df_index = df.index[frac_df.index[0]]
        df.loc[df_index, fraction] = '\\textbf{' + df.loc[df_index, fraction] + '}'
        # highlight the max f1
        frac_df = df[fraction].reset_index()
        # get all test_acc
        frac_df = frac_df[frac_df['metric'] == 'test_f1']
        frac_df['mean'] = frac_df[fraction].apply(lambda x: extract_mean(x)).astype(float)
        frac_df['stderr'] = frac_df[fraction].apply(lambda x: extract_var(x)).astype(float)
        # get the largest mean
        frac_df = frac_df[frac_df['mean'] == frac_df['mean'].max()]
        df_index = df.index[frac_df.index[0]]
        df.loc[df_index, fraction] = '\\textbf{' + df.loc[df_index, fraction] + '}'


        # highlight the max acc in our method
        frac_df = df[fraction].reset_index()
        # get all test_acc
        frac_df = frac_df[(frac_df['metric'] == 'test_acc') & (frac_df['selection'].str.contains("cov"))]
        frac_df['mean'] = frac_df[fraction].apply(lambda x: extract_mean(x)).astype(float)
        frac_df['stderr'] = frac_df[fraction].apply(lambda x: extract_var(x)).astype(float)
        # get the largest mean
        frac_df = frac_df[frac_df['mean'] == frac_df['mean'].max()]

        # # get the count of non-cov method
        df_index = frac_df.index[0]
        df_index = df.index[df_index]
        # # highlight
        df.loc[df_index, fraction] = '\\underline{' + df.loc[df_index, fraction] + '}'

        # highlight the max f1 in our method
        frac_df = df[fraction].reset_index()
        # get all test_acc
        frac_df = frac_df[(frac_df['metric'] == 'test_f1') & (frac_df['selection'].str.contains("cov"))]
        frac_df['mean'] = frac_df[fraction].apply(lambda x: extract_mean(x)).astype(float)
        frac_df['stderr'] = frac_df[fraction].apply(lambda x: extract_var(x)).astype(float)
        # get the largest mean
        frac_df = frac_df[frac_df['mean'] == frac_df['mean'].max()]
        df_index = frac_df.index[0]
        df_index = df.index[df_index]
        df.loc[df_index, fraction] = '\\underline{' + df.loc[df_index, fraction] + '}'
    return df

@hydra.main(config_name="base", version_base=None)
def dict_to_latex(cfg):
    # if any value is None, remove it recursively
    def remove_none(d):
        if isinstance(d, dict):
            return {k: remove_none(v) for k, v in d.items() if v is not None}
        elif isinstance(d, list):
            return [remove_none(v) for v in d]
        else:
            return d

    cfg = remove_none(cfg)
    print(cfg)
    expanded_rows = []
    for row in cfg.table.rows:
        print(row)
        values = row.values()
        print([type(v) for v in values])
        if any(isinstance(v, omegaconf.listconfig.ListConfig) for v in values):
            print(row)
            expanded_rows.extend(expand_configurations([row]))
        else:
            expanded_rows.append(row)
    print("------------")
    cfg.table.rows = expanded_rows
    print(cfg.table.rows)
    table = cfg.table
    table = OmegaConf.to_container(table, resolve=True)
    table = remove_none(table)


    run_calls = get_run_calls(table)

    ori_df = call(run_calls, cfg.stage)
    print(ori_df)

    ori_df.columns = [col.replace("__", ".") for col in ori_df.columns]
    # format
    df = copy.deepcopy(ori_df)
    # agrregate selection.XXX columns
    selection_cols = [col for col in df.columns if col.startswith("selection.")]
    # drop tags columns
    selection_cols = [col for col in selection_cols if not col.startswith("tags")]
    # drop fraction columns
    selection_cols = [col for col in selection_cols if not col.endswith("fraction")]

    if "WD" in cfg.table.name:
        selection_cols += ["finetuning.optimizer.feature_weight_decay"]
    # aggregate selection columns
    df["selection"] = df[selection_cols].apply(lambda x: x.dropna().to_dict(), axis=1)
    # now df selection is a dict I want "-".join keys
    df["selection"] = df["selection"].apply(lambda x: " + ".join([str(v) for k, v in x.items()]))
    # drop selection methods
    df = df.drop(columns=selection_cols)

    grouped_df = df.groupby(['dataset.name', 'backbone.name', 'selection', 'selection.fraction'])

    # Compute the mean and standard error for each group
    mean_df = grouped_df.mean()
    stderr_df = grouped_df.sem()

    # Format the mean and standard error
    formatted_df = mean_df.map(lambda x: f"{x:.4f}") + " ± " + stderr_df.map(lambda x: f"{x:.4f}")

    # Reset the index to make it a regular DataFrame
    formatted_df = formatted_df.reset_index()


    summary_keys = table["summary_keys"]
    logger.debug(f"Summary keys: {summary_keys}")
    # Pivot the table to make 'selection.fraction' columns and concatenate the metrics
    pivot_df = formatted_df.pivot_table(
        index=['dataset.name', 'backbone.name', 'selection'],
        columns='selection.fraction',
        values=summary_keys,
        aggfunc='first'
    )

    melted_df = pd.melt(
        formatted_df,
        id_vars=['dataset.name', 'backbone.name', 'selection.fraction', 'selection'],
        value_vars=summary_keys,
        var_name='metric',
        value_name='value'
    )

    # Pivot the table to have 'selection.fraction' columns
    pivot_df = melted_df.pivot_table(
        index=['dataset.name', 'backbone.name', 'selection', 'metric'],
        columns='selection.fraction',
        values='value',
        aggfunc='first'
    )
    print(pivot_df)
    # highlight the max value
    if "WD" in cfg.table.name or "SKMMv2" in cfg.table.name or "clip" in cfg.table.name.lower():
        logger.info("Not Highlighting max")
    else:
        pivot_df = highlight_max(pivot_df)

    table_name = cfg.table.name

    pivot_df.to_csv(f"./figures/{table_name}.csv")

    pivot_df.to_pickle(f"./figures/{table_name}.pkl")

    # to latex
    latex_table = pivot_df.to_latex()
    # change _ to \_
    latex_table = latex_table.replace("_", "\\_")
    with open(f"./figures/{table_name}.tex", "w") as f:
        f.write(latex_table)
    # copy to docs/paper/notes/figures/
    import shutil
    shutil.copy(f"./figures/{table_name}.tex", f"./docs/paper/notes/figures/{table_name}.tex")



if __name__ == "__main__":
    dict_to_latex()
