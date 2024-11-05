import hashlib
import copy
from omegaconf import OmegaConf
def get_cfg_hash(cfg_dict: dict) -> str:
    # if type(cfg_dict) == OmegaConf:
        # cfg_dict = OmegaConf.to_container(cfg_dict, resolve=True)
    temp_cfg = copy.deepcopy(cfg_dict)
    ignore_list = ["overwrite", "num_workers"]
    for k in ignore_list:
        if k in temp_cfg:
            del temp_cfg[k]
    cfg_hash = hashlib.md5(str(temp_cfg).encode()).hexdigest()
    return cfg_hash


def get_cfg_hash_without_fraction(cfg: OmegaConf) -> str:
    import hashlib
    from omegaconf import OmegaConf
    import copy
    # delete the selection.fraction
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if "selection" in cfg_dict:
        if "fraction" in cfg_dict["selection"]:
            del cfg_dict["selection"]["fraction"]
    temp_cfg = copy.deepcopy(cfg_dict)
    ignore_list = ["overwrite", "num_workers"]
    for k in ignore_list:
        if k in temp_cfg:
            del temp_cfg[k]
    cfg_hash = hashlib.md5(str(temp_cfg).encode()).hexdigest()
    return cfg_hash