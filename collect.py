import os
import hashlib
import wandb
import hydra
from omegaconf import OmegaConf
from utils.hash_utils import get_cfg_hash
from utils.minio_utils import get_minio_client, upload_directory, exists_on_minio, download_directory
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def check_file_exists(minio_client, hash_dir, file_name):
    return exists_on_minio(minio_client, hash_dir, file_name) or os.path.exists(f"{hash_dir}/{file_name}")

def check_dir_exists(minio_client, hash_dir):
    return exists_on_minio(minio_client, hash_dir) or os.path.exists(hash_dir)


def tosubmit(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_hash = get_cfg_hash(cfg_dict)
    os.makedirs("configs_tosubmit", exist_ok=True)
    os.system(f"cp ./configs/header.yaml ./configs_tosubmit/{cfg_hash}.yaml")
    with open(f"configs_tosubmit/{cfg_hash}.yaml", "a") as f:
        f.write(OmegaConf.to_yaml(cfg))
    return

def copy_to_temp_selection(cfg_hash, output_dir):
    hash_dir = f"outputs/finetuning/{cfg_hash}"
    wandb_summary_file = f"{hash_dir}/wandb/latest-run/files/wandb-summary.json"
    config_file = f"{hash_dir}/wandb/latest-run/files/config.yaml"
    os.system(f"cp {wandb_summary_file} ./{output_dir}/{cfg_hash}_wandb-summary.json")
    os.system(f"cp {config_file} ./{output_dir}/{cfg_hash}_config.yaml")

def save_config(cfg: OmegaConf):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_hash = get_cfg_hash(cfg_dict)
    cfg_file = f"temp/{cfg_hash}_config.yaml"
    with open(cfg_file, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

def collect_local(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_hash = get_cfg_hash(cfg_dict)
    # logger.info(f"f: {cfg.finetuning}")
    if hasattr(cfg, "finetuning"):
        dataset = cfg.dataset.name
        backbone = f"{cfg.backbone.name}-{cfg.backbone.version}"
        hash_dir = f"outputs/finetuning/{cfg_hash}"
        output_dir = f"collects/{dataset}/{backbone}/{cfg.selection.method}/{cfg.finetuning.optimizer.type}-{cfg.finetuning.layers}/c_conditioned={cfg.selection.c_conditioned}"
        os.makedirs(output_dir, exist_ok=True)
        from finetune import check_finished
    else:
        from data_select import check_finished
        hash_dir = f"outputs/selection/{cfg_hash}"
    if os.path.exists(hash_dir):
        if os.path.exists(hash_dir):
            if check_finished(hash_dir):
                print(f"Experiment with hash {cfg_hash} already exists and is finished.")
                save_config(cfg)
                copy_to_temp_selection(cfg_hash, output_dir)
            else:
                print(hash_dir)
                print(f"Experiment with hash {cfg_hash} already exists but is not finished.")
                tosubmit(cfg)
    else:
        logger.critical(f"{hash_dir} does not exist.")
        print(cfg)
        tosubmit(cfg)

def collect_wandb(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_hash = get_cfg_hash(cfg_dict)
    wandb_run_name = cfg_hash
    project = "data_pruning-finetuning"
    runs = wandb.Api().runs(f"{project}")
    runs = [run for run in runs if run.name == wandb_run_name]
    if len(runs) == 0:
        print(f"Run {wandb_run_name} not found.")
        return
    run = runs[0]
    hash_dir = f"outputs/finetuning/{cfg_hash}"
    os.makedirs(hash_dir, exist_ok=True)
    run_dir = f"{hash_dir}/wandb/latest-run/files"
    wandb_files = run.files()
    for file in wandb_files:
        file.download(root=run_dir, replace=True)
        logger.info(f"Downloaded {file.name} to {run_dir}/{file.name}")


@hydra.main(config_path="configs", config_name="default", version_base="1.3.0")
def collect(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_hash = get_cfg_hash(cfg_dict)
    wandb_files_dir = f"outputs/finetuning/{cfg_hash}/wandb/latest-run/files"
    if os.path.exists(wandb_files_dir):
        logger.debug(f"{wandb_files_dir} already exists.")
    else:
        collect_wandb(cfg)
    collect_local(cfg)

if __name__ == "__main__":
    collect()
