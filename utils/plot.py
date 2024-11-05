from collections import defaultdict
import itertools
import wandb
from omegaconf import OmegaConf
from hash_utils import get_cfg_hash

def get_cfg_from():
    return

cmd = "python opt.py -m dataset=cifar10 backbone=resnet50 selection=random selection.fraction=1000,2000,3000,4000 hydra/launcher=basic"

def get_cfg_from_cmd(cmd):
    cfg = {}
    for arg in cmd.split(" ")[1:]:
        key, value = arg.split("=")
        cfg[key] = value
    return cfg

def get_overrides_from_cmd(cmd):
    splits = cmd.split(" ")
    splits = [split for split in splits if "=" in split]
    # drop hydra/launcher=
    splits = [split for split in splits if "hydra/launcher" not in split]
    sweep_cfg = defaultdict(list)
    for i, split in enumerate(splits):
        key, value = split.split("=")
        sweep_cfg[key] = value.split(",")
    overrides_list = [dict(zip(sweep_cfg.keys(), values)) for values in itertools.product(*sweep_cfg.values())]
    return overrides_list

def get_cfg_from_cmd(cmd):
    overrides_list = get_overrides_from_cmd(cmd)
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    for overrides in overrides_list:
        with initialize(config_path="../configs", version_base="1.3.0"):
            cfg = compose(config_name="default", overrides=overrides)
            print(OmegaConf.to_yaml(cfg))

get_cfg_from_cmd(cmd)