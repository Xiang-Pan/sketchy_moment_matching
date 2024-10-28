import torch
import torch.nn as nn
import logging
from tqdm import tqdm
import hydra
import sys
import os
import torch.func as func
# from typing import str, int, bool, Optional

os.environ["HYDRA_FULL_ERROR"] = "1"
from utils.data_utils import get_raw_dataset_splits
from utils.model_utils import get_net

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# def get_layer_params(model, layer_names):
    # # Filter all modules to find ones with names matching layer_names
    # selected_params = []
    # for name, module in model.named_modules():
    #     if any(layer_name in name for layer_name in layer_names):
    #         selected_params.extend(list(module.parameters()))
    # return selected_params



def calculate_per_sample_grads(batch, batch_idx, model, loss_fn, output_dir, layer_names, num_classes, use_target, feature_scale):
    """_summary_: Here is the outputs grads
    """
    batch_size = batch[0].shape[0]
    data, targets = batch
    device = "cuda"
    data, targets = data.to(device=device), targets.to(device=device)
    model.zero_grad()

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    for name, param in model.named_parameters():
        param.requires_grad = any(layer_name in name for layer_name in layer_names)


    params = [p for p in model.parameters() if p.requires_grad]
    if batch_idx == 0:
        logger.debug(f"layer_names: {layer_names}")
        for p in params:
            logger.debug(f"p.shape: {p.shape}")

    if layer_names == ["all"]:
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
    else:
        params = {k: v for k, v in model.named_parameters() if any(layer_name in k for layer_name in layer_names)}
        buffers = {k: v for k, v in model.named_buffers() if any(layer_name in k for layer_name in layer_names)}
    # we assume the params are in the same order as the layer_names
    logger.debug(f"params: {params}")
    cls_name = layer_names[-1]
    pre_sample_grads = calculate_per_sample_grads_from_output(batch, model, params, buffers, num_classes, use_target=use_target, cls_name=cls_name, feature_scale=feature_scale)
    return pre_sample_grads



def calculate_per_sample_grads_from_output(batch, model, params, buffers, num_classes, use_target, cls_name, feature_scale):
    # sourcery skip: merge-comparisons, remove-unreachable-code
    data, targets = batch
    if use_target == "random":
        logger.critical("Using random targets")
        targets = torch.randint(0, num_classes, (data.shape[0],), device='cuda')
    def get_preds(params, buffers, sample, target, logit_idx):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)
        predictions = func.functional_call(model, (params, buffers), (batch,))
        loss = predictions.squeeze(0)[logit_idx]
        return loss
    def get_loss(params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)
        predictions = func.functional_call(model, (params, buffers), (batch,))
        loss = nn.CrossEntropyLoss()(predictions, targets)
        return loss

    if use_target == "random" or use_target == True:
        compute_loss = get_loss
        ft_compute_grad = func.grad(compute_loss)
        ft_compute_sample_grad = func.vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
        ft_per_sample_grads = ft_compute_sample_grad(params, buffers, data, targets)
        for k, v in ft_per_sample_grads.items():
            if cls_name not in k:
                ft_per_sample_grads[k] = v * feature_scale
        per_sample_grads = [v.view((data.shape[0], -1)) for k, v in ft_per_sample_grads.items() if v is not None]
        per_sample_grads = torch.cat(per_sample_grads, dim=1)
    else:
        raise NotImplementedError
        sample_grads = []
        for logit_idx in tqdm(range(num_classes), desc="calculate_per_sample_grads_from_output"):
            from functools import partial
            compute_loss = partial(get_preds, logit_idx=logit_idx)
            ft_compute_grad = func.grad(compute_loss)
            ft_compute_sample_grad = func.vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
            ft_per_sample_grads = ft_compute_sample_grad(params, buffers, data, targets)
            per_sample_grads = [v.view((data.shape[0], -1)) for k, v in ft_per_sample_grads.items() if v is not None]
            logit_grads = torch.cat(per_sample_grads, dim=1).detach().cpu()
            sample_grads.append(logit_grads)
        per_sample_grads = torch.cat(sample_grads, dim=1)
    return per_sample_grads

def calculate_per_sample_grads_from_loss():
    pass


def construct_G(dataset, model, layer_names, output_dir, batch_size, sketching_dim, num_classes, use_target, feature_scale):
    k = sketching_dim
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch_nums = len(dataloader)
    if all(
        os.path.exists(
            f"{output_dir}/sketching_dim={sketching_dim}/batch-idx={batch_idx}_device-id=0.pt"
        )
        for batch_idx in range(batch_nums)
    ):
        logger.info("All batches already exist. Skipping...")
    else:
        missing_list = [batch_idx for batch_idx in range(batch_nums) if not os.path.exists(f"{output_dir}/batch-idx={batch_idx}_device-id=0.pt")]
        logger.info(f"Missing batches: {missing_list}")
        os.makedirs(f"{output_dir}/sketching_dim={sketching_dim}", exist_ok=True)
        for batch_idx, (data, targets) in enumerate(tqdm(dataloader, desc="construct_G")):
            if os.path.exists(f"{output_dir}/sketching_dim={sketching_dim}/batch-idx={batch_idx}_device-id=0.pt"):
                logger.info(f"batch-idx={batch_idx} already exists. Skipping...")
                continue
            data, targets = data.to(device='cuda'), targets.to(device='cuda')
            per_sample_grads = calculate_per_sample_grads((data, targets), batch_idx, model, nn.CrossEntropyLoss(), output_dir, layer_names, num_classes=num_classes, use_target=use_target, feature_scale=feature_scale)
            ori_dim = per_sample_grads.shape[1]
            chunk_num = 1 if use_target else 20
            chunk_size = ori_dim // chunk_num
            sketch_list = []
            for chunk_idx in range(chunk_num):
                S = torch.randn((chunk_size, k), device='cuda').normal_(mean=0, std=(1/k)**0.5)
                sketched_grads = torch.matmul(per_sample_grads[:, chunk_idx*chunk_size:(chunk_idx+1)*chunk_size], S).detach().cpu()
                sketch_list.append(sketched_grads)
            sketched_grads = torch.stack(sketch_list, dim=1)
            sketched_grads = torch.sum(sketched_grads, dim=1)
            # save it to disk
            torch.save(sketched_grads, f"{output_dir}/sketching_dim={sketching_dim}/batch-idx={batch_idx}_device-id=0.pt")
        sketch_list = []
        for batch_idx in range(batch_nums):
            sketched_grads = torch.load(f"{output_dir}/sketching_dim={sketching_dim}/batch-idx={batch_idx}_device-id=0.pt")
            sketch_list.append(sketched_grads)
        sketch_list = torch.cat(sketch_list, dim=0)
        torch.save(sketch_list, f"{output_dir}/sketching_dim={sketching_dim}.pt")


def get_layer_names(backbone_name, layers):
    # ---
    # last two layers
    if layers == "all":
        layer_names = ["all"]
    elif layers == -1:
        if backbone_name in ["clip-vit-base-patch32", "tinynet_e"]:
            layer_names = ["classification_head"]
        elif backbone_name == "resnet18":
            layer_names = ["fc"]
        elif backbone_name == "resnet50":
            layer_names = ["fc"]
    elif layers == -2:
        if backbone_name in ["clip-vit-base-patch32", "tinynet_e"]:
            layer_names = ["image_encoder.visual.visual_projection"] + ["classification_head"]
        elif backbone_name == "resnet18":
            layer_names = ["layer4.1.conv2"] + ["fc"]
        elif backbone_name == "resnet50":
            layer_names = ["layer4.2.conv3"] + ["fc"]
    elif layers == -3:
        if backbone_name in ["clip-vit-base-patch32"]:
            layer_names = ["image_encoder.visual.transformer.resblocks.11"] + ["image_encoder.visual.visual_projection"] + ["classification_head"]
        else:
            raise ValueError(f"Unknown backbone_name: {backbone_name}")
    else:
        raise ValueError(f"Unknown layers: {layers}")
    return layer_names

def get_batch_size(backbone_name):
    #* HARD CODED
    if backbone_name == "resnet18":
        batch_size = 128
    elif backbone_name == "resnet50":
        batch_size = 128
    elif backbone_name == "clip-vit-base-patch32":
        batch_size = 32
    elif backbone_name == "tinyclip":
        batch_size = 32
    elif backbone_name == "tinynet_e":
        batch_size = 128
    else:
        raise ValueError(f"Unknown backbone_name: {backbone_name}")
    return batch_size

def get_grads(dataset_name,
                backbone_name,
                backbone_version,
                sketching_dim,
                layers,
                use_target=False,
                feature_scale=1.0,
                cls_pretrain_size=None,
                ):
    """
    the output_dir is constructed from the input arguments
        dataset_name: str
        backbone_name: str
        backbone_version: str
        sketching_dim: int
        layers: str
        use_target: bool
        feature_scale[Optional]: float
        cls_pretrain_size[Optional]: str
    """
    layer_names = get_layer_names(backbone_name, layers=layers)
    batch_size = get_batch_size(backbone_name)
    layer_names_str = "-".join(layer_names)
    output_dir = f"cached_grads/dataset={dataset_name}/backbone={backbone_name}-version={backbone_version}/layer_names={layer_names_str}/batch_size={batch_size}/use_target={use_target}"
    if feature_scale != 1.0:
        logger.info(f"feature_scale: {feature_scale}")
        output_dir = f"{output_dir}/feature_scale={feature_scale}"
    if cls_pretrain_size is not None:
        output_dir = f"{output_dir}/cls_pretrain_size={cls_pretrain_size}"
    grads_file = f"{output_dir}/sketching_dim={sketching_dim}.pt"
    grads = torch.load(grads_file)
    return grads

@hydra.main(config_path="configs", config_name="default", version_base="1.3.0")
def main(cfg):  # sourcery skip: remove-unnecessary-else, swap-if-else-branches
    model = get_net(cfg, cfg.dataset.num_classes).cuda()
    logger.info(model)
    backbone_name = cfg.backbone.name
    layer_names = get_layer_names(backbone_name, layers=cfg.layers)
    batch_size = get_batch_size(backbone_name)
    layer_names_str = "-".join(layer_names)
    output_dir = f"cached_grads/dataset={cfg.dataset.name}/backbone={cfg.backbone.name}-version={cfg.backbone.version}/layer_names={layer_names_str}/batch_size={batch_size}/use_target={cfg.use_target}"
    if hasattr(cfg, "feature_scale"):
        output_dir = f"{output_dir}/feature_scale={cfg.feature_scale}"
        feature_scale = cfg.feature_scale
    else:
        feature_scale = 1.0

    if hasattr(cfg, "cls_pretrain_size"):
        output_dir = f"{output_dir}/cls_pretrain_size={cfg.cls_pretrain_size}"


    os.makedirs(output_dir, exist_ok=True)
    train_dataset, val_dataset, test_dataset = get_raw_dataset_splits(cfg=cfg, all_test_transform=True)


    if hasattr(cfg, "cls_pretrain_size"):
        logger.info(f"cls_pretrain_size: {cfg.cls_pretrain_size}")
        cls_pretrain_size = cfg.cls_pretrain_size
        if cls_pretrain_size in ["full", "full-1"]:
            #! equivalent to cls_pretrain_size = "full-1"
            logger.info("cls_pretrain_size: full")
            cls_pretrain_size = len(train_dataset)
            from linear_probe import test_eb
            from utils.data_utils import load_eb_dataset_cfg
            eb_dataset_dict = load_eb_dataset_cfg(cfg, device="cuda")
            idxes = list(range(len(train_dataset)))
            idxes = torch.Tensor(idxes).long()
            logger.info(f"idxes: {idxes}")
            # disable wandb
            import wandb
            wandb.init(mode="disabled")
            lp = test_eb(
                    dataset_dict=eb_dataset_dict,
                    idxes=idxes,
                    weights=None,
                    use_weights=False,
                    seed=0,
                    dataset_name=cfg.dataset.name,
                    name="full",
                    use_mlp=False,
                    task="classification",
                    test_as_val=True,
                    epoch=-1,
                    tune=False,
                )
            logger.info(f"lp: {lp}")
            # lp is a pipeline object
            clf = lp.named_steps['clf']
            weight = clf.coef_
            bias = clf.intercept_
            logger.debug(f"weight.shape: {weight.shape}, bias.shape: {bias.shape}")
            model.classification_head.weight.data = torch.Tensor(weight).cuda()
            model.classification_head.bias.data = torch.Tensor(bias).cuda()
            # set the lp weights to the model
            model.classification_head
        elif cls_pretrain_size == "full-2":
            ckpt_path = "./outputs/finetuning/83180a0be01139253378dc20968cc2ca/last.ckpt"
            ckpt = torch.load(ckpt_path)
            state_dict = ckpt["state_dict"]
            new_state_dict = {
                k.replace("net.", ""): state_dict[k]
                for k in list(state_dict.keys())
                if "net." in k
            }
            model.load_state_dict(new_state_dict)
            model.eval()
        else:
            raise ValueError(f"Unknown cls_pretrain_size: {cls_pretrain_size}")


    G = construct_G(train_dataset, model, layer_names, output_dir, batch_size=batch_size, sketching_dim=cfg.sketching_dim, num_classes=cfg.dataset.num_classes, use_target=cfg.use_target, feature_scale=feature_scale)

if __name__ == "__main__":
    main()