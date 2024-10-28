import hydra
import torch
import copy
import os
import json
import torch.nn as nn
import numpy as np

import boto3
os.environ['AWS_ACCESS_KEY_ID'] = 'HmBR4sWSV5ukyVIZfmwA'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'gRI4CqFF6XCDdjXpyxJcwUMFx3bnsC3LKIeE8sl9'
os.environ['AWS_ENDPOINT_URL'] = 'http://minioapi.xiangpan.site'
# client = boto3.client('s3')

import wandb
wandb.require('core')

from omegaconf import OmegaConf
from utils.hash_utils import get_cfg_hash, get_cfg_hash_without_fraction
from pytorch_lightning import seed_everything
from utils.data_utils import get_raw_dataset_splits
from pytorch_lightning import Trainer
from pytorch_lightning import LightningDataModule
from models.litclassifier import LitClassifier
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateFinder
from utils.hash_utils import get_cfg_hash, get_cfg_hash_without_fraction

import logging
logging.basicConfig(level=logging.DEBUG, filename="outputs/finetuning.log", filemode="a")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


#* clean this
class MyDataModule(LightningDataModule):
    def __init__(self,
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    batch_size
                ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = 2
        self.pin_memory = True

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=1024,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1024,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

def check_finished(cfg: OmegaConf) -> bool:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_hash = get_cfg_hash(cfg_dict)
    finetuning_dir = f"outputs/finetuning/{cfg_hash}"
    return check_finished(hash_dir=finetuning_dir)

def check_finished(hash_dir: str) -> bool:
    keys = ["test/acc_epoch"]
    results_path = f"{hash_dir}/wandb/latest-run/files/wandb-summary.json"
    if not os.path.exists(results_path):
        logger.info(f"Results file {results_path} does not exist.")
        return False
    with open(results_path, "r") as f:
        results = json.load(f)
    for key in keys:
        if key not in results:
            logger.critical(f"{key} not in {results_path}")
            return False
    return True


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)

def finetune(cfg, train_index):
    train_dataset, val_dataset, test_dataset = get_raw_dataset_splits(cfg, transform=None, train_index=train_index)
    logger.info(f"train_dataset: {len(train_dataset)}")
    logger.info(f"val_dataset: {len(val_dataset)}")
    logger.info(f"test_dataset: {len(test_dataset)}")
    logger.info(f"train_index: {len(train_index)}")
    assert len(train_dataset) == len(train_index)
    # test_as_val
    val_dataset = test_dataset
    datamodule = MyDataModule(train_dataset, val_dataset, test_dataset, cfg.finetuning.batch_size)
    module = LitClassifier(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_hash = get_cfg_hash(cfg_dict)
    from pytorch_lightning.loggers import WandbLogger
    wandb_logger = WandbLogger(project="data_pruning-finetuning",
                               name=cfg_hash,
                               dir=f"outputs/finetuning/{cfg_hash}")
    # checkpoint
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc",
        # dirpath="s3://labs/data_pruning/outputs/finetuning/{cfg_hash}",
        dirpath=f"outputs/finetuning/{cfg_hash}",
        save_top_k=1,
        save_last=True,
        mode="max",
        auto_insert_metric_name=False,
        filename="epoch={epoch}-val_acc={val/acc:.4f}",
        enable_version_counter=False,
    )
    trainer = Trainer(
                        default_root_dir=f"outputs/finetuning/{cfg_hash}",
                        max_epochs=cfg.finetuning.max_epochs,
                        accelerator="gpu",
                        logger=[wandb_logger],
                        callbacks=[checkpoint_callback],
                        log_every_n_steps=1,
                        check_val_every_n_epoch=min(10, cfg.finetuning.max_epochs),
    )
    trainer.fit(module, datamodule)
    results = trainer.test(ckpt_path="best", datamodule=datamodule)
    logger.info(f"Results: {results}")
    logger.info(f"Best model: {checkpoint_callback.best_model_path}")
    with open(f"outputs/finetuning/{cfg_hash}/results.json", "w") as f:
        json.dump(results[0], f)
    with open(f"outputs/finetuning/{cfg_hash}/best_model.txt", "w") as f:
        f.write(checkpoint_callback.best_model_path)
    logger.critical(f"outputs/finetuning/{cfg_hash}")
    logger.critical(f"outputs/finetuning/{cfg_hash}/results.json")
    logger.critical(f"wandb: {wandb_logger.experiment.url}")
    wandb.finish()
    # from utils.storage_utils import upload_folder
    # upload_folder(f"outputs/finetuning/{cfg_hash}")


def sklearn_linear_probe(cfg, index_dir):
    from linear_probe import linear_probe
    from utils.data_utils import load_eb_dataset_cfg
    eb_dataset_dict = load_eb_dataset_cfg(cfg)
    c_conditioned_idxes = torch.load(f"{index_dir}/c_conditioned_idxes.pt")
    c_conditioned_weights = None
    c_unconditioned_idxes = torch.load(f"{index_dir}/c_unconditioned_idxes.pt")
    c_unconditioned_weights = None
    linear_probe(c_unconditioned_idxes=c_unconditioned_idxes,
                    c_unconditioned_weights=c_unconditioned_weights,
                    c_conditioned_idxes=c_conditioned_idxes,
                    c_conditioned_weights=c_conditioned_weights,
                    dataset_dict=eb_dataset_dict,
                    seed=cfg.seed)

def get_train_index(cfg, index_dir):
    if cfg.selection.c_conditioned == 1:
        logger.info(f"Loading selection from {index_dir}")
        c_conditioned_idxes = torch.load(f"{index_dir}/c_conditioned_idxes.pt")
        train_index = c_conditioned_idxes
    elif cfg.selection.c_conditioned == 0:
        c_unconditioned_idxes = torch.load(f"{index_dir}/c_unconditioned_idxes.pt")
        train_index = c_unconditioned_idxes
    elif isinstance(cfg.selection.c_conditioned, float):
        logger.info(f"Class conditioned selection {cfg.selection.c_conditioned}")
        s = torch.load(f"{index_dir}/s.pt", map_location="cpu")
        train_dataset, val_dataset, test_dataset = get_raw_dataset_splits(cfg, transform=None, train_index=None)
        y = torch.tensor(train_dataset.targets)
        m = cfg.selection.fraction
        if isinstance(m, float):
            m = int(m * len(train_dataset))
        c_conditioned_m = int(cfg.selection.c_conditioned * m)
        unique_classes = torch.unique(y)
        logger.debug(f"unique_classes: {unique_classes}")
        top_m_index = []
        for c in unique_classes:
            c_m = c_conditioned_m // len(unique_classes)
            idxes = torch.where(y == c)[0]
            s_c = s[idxes]
            logger.debug(f"c_m: {c_m}")
            _, top_m_c = torch.topk(s_c, c_m, largest=True)
            s_idxes = idxes.squeeze()[top_m_c].cpu()
            top_m_index.append(s_idxes)
        top_m_index = torch.cat(top_m_index)
        c_conditioned_index = top_m_index
        # class unconditioned select the c_conditioned * fraction of data
        # remove the top_m_index
        full_index = torch.arange(len(train_dataset))
        candidates = torch.where(~torch.isin(full_index, top_m_index))[0]
        c_unconditioned_m = m - c_conditioned_m
        _, c_unconditioned_candidates = torch.topk(s[candidates], c_unconditioned_m, largest=True)
        c_unconditioned_index = candidates[c_unconditioned_candidates]
        train_index = torch.cat([c_conditioned_index, c_unconditioned_index])
        logger.info(f"train_index: {len(train_index)}")
        logger.info(f"c_conditioned_m: {len(c_conditioned_index)}")
        logger.info(f"c_unconditioned_m: {len(c_unconditioned_index)}")
    else:
        raise ValueError(f"Unknown c_conditioned: {cfg.selection.c_conditioned}")
    return train_index

@hydra.main(config_path="configs", config_name="default", version_base="1.3.0")
def main(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_hash = get_cfg_hash(cfg_dict)

    selection_cfg_dict = copy.deepcopy(cfg_dict)
    if "finetuning" in selection_cfg_dict:
        del selection_cfg_dict["finetuning"]
    if "c_conditioned" in selection_cfg_dict["selection"]:
        del selection_cfg_dict["selection"]["c_conditioned"]
    selection_cfg_hash = get_cfg_hash(selection_cfg_dict)
    logger.info(f"selection_cfg_hash: {selection_cfg_hash}")

    selection_cfg = copy.deepcopy(cfg)
    selection_cfg_dict = OmegaConf.to_container(selection_cfg, resolve=True)
    if "finetuning" in selection_cfg_dict:
        del selection_cfg_dict["finetuning"]
    if "c_conditioned" in selection_cfg_dict["selection"]:
        del selection_cfg_dict["selection"]["c_conditioned"]
    if "mixselection_ratio" in selection_cfg_dict["selection"]:
        del selection_cfg_dict["selection"]["mixselection_ratio"]
        del selection_cfg_dict["selection"]["mixselection_method"]
        del selection_cfg_dict["selection"]["mixselection_mixmethod"]
    if "finetuning" in selection_cfg_dict:
        del selection_cfg_dict["finetuning"]
    selection_cfg_hash = get_cfg_hash(selection_cfg_dict)
    selection_cfg = OmegaConf.create(selection_cfg_dict)
    selection_cfg_hash_without_fraction = get_cfg_hash_without_fraction(selection_cfg)


    if check_finished(cfg) and not cfg.overwrite:
        logger.info("Finetuning already finished")
        return
    logger.info(f"Config: \n {cfg}")
    seed_everything(cfg.selection.seed)
    # load train index from DeepCore
    if cfg.selection.method in ["Uniform", "ContextualDiversity", "Glister", "GraNd", "Herding", "Forgetting", "DeepFool", "Uncertainty-Entropy", "Uncertainty-Margin", "Uncertainty-LeastConfidence"]:
        # Normal train index
        from methods.deepcore_methods import deepcore_load
        dataset = cfg.dataset.name
        backbone = cfg.backbone.name
        method = cfg.selection.method
        m = cfg.selection.fraction
        test_as_val = True
        indexes, weights = deepcore_load(dataset=dataset,
                                         backbone=backbone,
                                         method=method,
                                         m=m,
                                         seed=cfg.seed,
                                         test_as_val=test_as_val,
                                         layers=cfg.finetuning.layers,
                                         c_conditioned=cfg.selection.c_conditioned)
        train_index = indexes
    elif cfg.selection.method == "cov_ntk_perclass":
        selection_root_dir = f"outputs/selection"

        selection_cfg = OmegaConf.create(selection_cfg_dict)
        selection_cfg_hash = get_cfg_hash(selection_cfg_dict)
        index_dir = f"{selection_root_dir}/{selection_cfg_hash}"
        if cfg.selection.c_conditioned == 1:
            c_conditioned_idxes = torch.load(f"{index_dir}/c_conditioned_idxes.pt")
            train_index = c_conditioned_idxes
        elif cfg.selection.c_conditioned == 0:
            c_unconditioned_idxes = torch.load(f"{index_dir}/c_unconditioned_idxes.pt")
            train_index = c_unconditioned_idxes
        elif cfg.selection.c_conditioned == "sampled":
            c_sampled_idxes = torch.load(f"{index_dir}/c_sampled_idxes.pt")
            train_index = c_sampled_idxes
        else:
            raise ValueError(f"Unknown c_conditioned: {cfg.selection.c_conditioned}")
        train_index = train_index.detach().cpu()
    else:
        selection_root_dir = "outputs/selection"

        selection_cfg = OmegaConf.create(selection_cfg_dict)
        m = cfg.selection.fraction
        if cfg.selection.method == "full":
            # from utils.data_utils import load_eb_dataset_cfg
            # eb_dataset_dict = load_eb_dataset_cfg(cfg)
            train_dataset, val_dataset, test_dataset = get_raw_dataset_splits(cfg, transform=None, train_index=None)
            train_index = torch.arange(len(train_dataset))
        elif cfg.selection.method == "random":
            from utils.data_utils import load_eb_dataset_cfg
            eb_dataset_dict = load_eb_dataset_cfg(cfg)
            train_X = eb_dataset_dict["train"]["X"]
            train_y = eb_dataset_dict["train"]["Y"]
            logger.info(f"Class conditioned selection {cfg.selection.c_conditioned}")
            if cfg.selection.c_conditioned:
                num_classes = cfg.dataset.num_classes
                c_m = m // num_classes
                for c in range(num_classes):
                    idxes = torch.where(train_y == c)[0]
                    c_idxes = np.random.choice(idxes, c_m, replace=False)
                    if c == 0:
                        train_index = c_idxes
                    else:
                        train_index = np.concatenate([train_index, c_idxes])
            else:
                train_index = np.random.choice(len(train_dataset), m, replace=False)
            # selection_cfg_hash = get_cfg_hash(selection_cfg_dict)
            # index_dir = f"{selection_root_dir}/{selection_cfg_hash}"
            # files = os.listdir(index_dir)
            # logger.debug(f"Files in {index_dir}: {files}")
            # train_index = get_train_index(cfg, index_dir)
        elif cfg.selection.method == "SkMMv2":
            cfg_hash = get_cfg_hash(selection_cfg)
            output_dir = f"{selection_root_dir}/{cfg_hash}"
            c_unconditioned_idxes = torch.load(f"{output_dir}/c_unconditioned_idxes.pt", map_location="cpu")
            train_index = c_unconditioned_idxes
        else:
            s_cfg_hash = get_cfg_hash_without_fraction(selection_cfg)
            s_output_dir = f"{selection_root_dir}/{selection_cfg_hash_without_fraction}"
            output_dir = f"{selection_root_dir}/{selection_cfg_hash}"
            s = torch.load(f"{s_output_dir}/s.pt", map_location="cpu")
            train_dataset, val_dataset, test_dataset = get_raw_dataset_splits(cfg, transform=None, train_index=None)
            from utils.data_utils import load_eb_dataset_cfg
            eb_dataset_dict = load_eb_dataset_cfg(cfg)
            train_y = eb_dataset_dict["train"]["Y"]
            unique_classes = torch.unique(train_y)
            logger.debug(f"unique_classes: {unique_classes}")
            m = cfg.selection.fraction
            if isinstance(m, float):
                m = int(m * len(train_dataset))

            if hasattr(cfg.selection, "preselection"):
                match cfg.selection.c_conditioned:
                    case "sampled":
                        c_sampled_idxes = torch.load(f"{output_dir}/c_sampled_idxes.pt")
                        train_index = c_sampled_idxes
                    case True:
                        c_conditioned_idxes = torch.load(f"{output_dir}/c_conditioned_idxes.pt")
                        train_index = c_conditioned_idxes
                    case False:
                        c_unconditioned_idxes = torch.load(f"{output_dir}/c_unconditioned_idxes.pt")
                        train_index = c_unconditioned_idxes
                    case _:
                        raise ValueError(f"Unknown preselection: {cfg.selection.preselection}")
            elif hasattr(cfg.selection, "mixselection_ratio"):
                from methods.deepcore_methods import deepcore_load
                assert cfg.selection.c_conditioned != 1, "Cannot mixselection with c_conditioned"

                mixselection_method = cfg.selection.mixselection_method
                mixselection_mixmethod = cfg.selection.mixselection_mixmethod
                mixselection_ratio = cfg.selection.mixselection_ratio
                assert cfg.selection.mixselection_ratio is not None
                assert cfg.selection.mixselection_method is not None
                assert cfg.selection.mixselection_mixmethod is not None

                name_A, method_A, m_A = mixselection_method.split("_")
                assert name_A == "deepcore"
                test_as_val = True
                layers = cfg.finetuning.layers
                dataset = cfg.dataset.name
                backbone = cfg.backbone.name
                m_A = int(m_A)
                mixselection_idxes, mixselection_weights = deepcore_load(dataset=dataset,
                                                                        backbone=backbone,
                                                                        method=method_A,
                                                                        m=m_A,
                                                                        seed=cfg.seed,
                                                                        test_as_val=test_as_val, layers=layers)
                if mixselection_mixmethod == "sampling":
                    mixselection_ratio = cfg.selection.mixselection_ratio
                    dist_1 = mixselection_weights
                    dist_2 = s
                    mixselection_weights = mixselection_ratio * dist_1 + (1 - mixselection_ratio) * dist_2
                    c_unconditioned_idxes = np.random.choice(np.arange(len(s)), m, p=mixselection_weights)
                    c_unconditioned_weights = mixselection_weights[c_unconditioned_idxes]
                elif mixselection_mixmethod == "concat":
                    assert cfg.selection.c_conditioned == 0, "Cannot mixselection with c_conditioned"
                    m1 = int(m * cfg.selection.mixselection_ratio)
                    m2 = m - m1
                    from methods.cov_opt import select_top_m
                    logger.debug(f"mixselection_weights: {mixselection_weights}")
                    c_unconditioned_idxes_1 = mixselection_idxes[:m1]
                    # c_unconditioned_idxes_2, c_unconditioned_weights_2 = select_top_m(s, m2, y=None, class_conditioned=False)
                    # c_unconditioned_idxes = np.concatenate([c_unconditioned_idxes_1, c_unconditioned_idxes_2])
                    # get the samples not in c_unconditioned_idxes_1
                    # set s to be -inf for c_unconditioned_idxes_1
                    s = s.clone()
                    # inf_tensor = -np.inf * torch.ones_like(s)
                    # s[c_unconditioned_idxes_1] = inf_tensor
                    for idx in c_unconditioned_idxes_1:
                        s[idx] = -np.inf
                    c_unconditioned_idxes_2, c_unconditioned_weights_2 = select_top_m(s, m2, y=None, class_conditioned=False)
                    c_unconditioned_idxes = np.concatenate([c_unconditioned_idxes_1, c_unconditioned_idxes_2])

                else:
                    raise ValueError(f"Unknown mixselection_mixmethod: {mixselection_mixmethod}")
                train_index = c_unconditioned_idxes

            elif cfg.selection.c_conditioned == 1:
                train_index_list = []
                for c in unique_classes:
                    idxes = torch.where(train_y == c)[0]
                    s_c = s[idxes]
                    # select top m / num_classes
                    c_m = m // len(unique_classes)
                    _, top_m_c = torch.topk(s_c, c_m, largest=True)
                    logger.debug(f"top_m_c: {top_m_c}")
                    s_idxes = idxes.squeeze()[top_m_c].cpu()
                    train_index_list.append(s_idxes)
                train_index = torch.cat(train_index_list)
                logger.info(f"train_index: {len(train_index)}")
            elif cfg.selection.c_conditioned == 0:
                candidates = torch.arange(len(train_dataset))
                _, top_m_candidates = torch.topk(s, m, largest=True)
                train_index = candidates[top_m_candidates]
            elif cfg.selection.c_conditioned == "sampled":
                logger.info(f"cfg.selection.c_conditioned={cfg.selection.c_conditioned}")
                s = torch.softmax(s, dim=0)
                train_index = torch.multinomial(s, m, replacement=False)
            else:
                raise ValueError(f"Unknown c_conditioned: {cfg.selection.c_conditioned}")
    wandb.init(project="data_pruning-finetuning", name=cfg_hash, config=cfg_dict)
    finetune(cfg, train_index)
    wandb.finish()

if __name__ == "__main__":
    main()