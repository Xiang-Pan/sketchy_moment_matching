import os
import torch
from omegaconf import OmegaConf
from logging import getLogger
from torchvision import transforms
from hydra.utils import instantiate
import numpy as np
import torchvision
from joblib import Memory
from hydra import initialize, compose
logger = getLogger(__name__)
import sys
sys.path.append("/vast/xp2030/labs/data_pruning")
from task_datasets.utk import UTKDataset
from task_datasets.cardio import CardioDataset
memory = Memory("./cache")
from clip import clip
from PIL import Image

def get_transforms(cfg, phase="train", all_test_transform=False):
    if cfg.backbone.name == "clip-vit-base-patch32":
        _, preprocess = clip.load("ViT-B/32", device="cuda", jit=False)
        transform = preprocess
        logger.critical(f"Using CLIP transform: {transform}")
        return transform
    elif cfg.backbone.name == "tinyclip":
        import timm
        backbone_version = cfg.backbone.version
        num_classes = cfg.dataset.num_classes
        model = timm.create_model(backbone_version, pretrained=True, num_classes=num_classes)
        data_config = timm.data.resolve_model_data_config(model)
        is_training = True if phase == "train" else False
        if all_test_transform:
            is_training = False
        transform = timm.data.create_transform(**data_config, is_training=is_training)
        logger.critical(f"Using TinyCLIP transform: {transform}")
        return transform
    elif cfg.backbone.name == "tinynet_e":
        import timm
        backbone_version = cfg.backbone.version
        num_classes = cfg.dataset.num_classes
        model = timm.create_model(backbone_version, pretrained=True, num_classes=num_classes)
        data_config = timm.data.resolve_model_data_config(model)
        is_training = True if phase == "train" else False
        if all_test_transform:
            is_training = False
        transform = timm.data.create_transform(**data_config, is_training=is_training)
        logger.critical(f"Using TinyNet-E transform: {transform}")
        return transform
    elif cfg.backbone.name == "vit_tiny_patch16_224":
        import timm
        backbone_version = cfg.backbone.version
        num_classes = cfg.dataset.num_classes
        model = timm.create_model(backbone_version, pretrained=True, num_classes=num_classes)
        data_config = timm.data.resolve_model_data_config(model)
        is_training = True if phase == "train" else False
        if all_test_transform:
            is_training = False
        transform = timm.data.create_transform(**data_config, is_training=is_training)
    elif cfg.backbone.name == "resnet18" or cfg.backbone.name == "resnet50":
        transform_train = transforms.Compose([
            transforms.Resize(int(224 * (8 / 7)), interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.Resize(int(224 * (8 / 7)), interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        if all_test_transform:
            transform_train = transform_test
        if phase == "train":
            transform = transform_train
        elif phase == "val":
            transform = transform_test
        elif phase == "train_val":
            transform = None
        elif phase == "test":
            transform = transform_test
        else:
            raise ValueError(f"Unknown phase {phase}")
    else:
        raise ValueError(f"Unknown backbone name: {cfg.backbone.name}")
    return transform

class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
    
    
def binary_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    train_dataset = torchvision.datasets.MNIST(
        root=os.environ["DATAROOT"],
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = torchvision.datasets.MNIST(
        root=os.environ["DATAROOT"],
        train=False,
        download=True,
        transform=transform,
    )
    train_dataset.targets = (train_dataset.targets == 0).float()
    test_dataset.targets = (test_dataset.targets == 0).float()
    return train_dataset, test_dataset


def get_raw_dataset(cfg,
                    phase="train",
                    transform=None,
                    all_test_transform=False,
                    debug=False,
                    ) -> torch.utils.data.Dataset:
    
    if transform is None:
        transform = get_transforms(cfg,
                                   phase=phase,
                                   all_test_transform=all_test_transform)
    else:
        transform = transform
    logger.debug(f"transform: {transform}")
    logger.debug(f"phase: {phase}")
    logger.debug(f"transform: {transform}")
    if cfg.dataset.name == "CelebA":
        logger.info(f"cfg.dataset: {cfg.dataset}")
        split_remap = {
            "train": "train",
            "val": "valid",
            "test": "test",
        }
        # Blond_Hair
        attr_names = cfg.dataset.attr_names
        dataset = torchvision.datasets.CelebA(
            root=os.environ["DATAROOT"],
            split=split_remap[phase],
            target_type="attr",
            transform=transform,
            download=True,
        )
        idx_list = []
        attr_idx_0 = dataset.attr_names.index(attr_names[0])
        attr_idx_1 = dataset.attr_names.index(attr_names[1])
        target_transform = transforms.Lambda(lambda x: 1 * x[attr_idx_0] + 2 * x[attr_idx_1])
        dataset.target_transform = target_transform
        return dataset
    elif cfg.dataset.name == "StanfordCars":
        #* Following the error to download the dataset
        dataset = torchvision.datasets.StanfordCars(
            root=os.environ["DATAROOT"],
            split="train" if phase in ["train", "trainval"] else "test",
            transform=transform,
            download=False,
        )
        return dataset
    else:
        dataset_name = cfg.dataset.name
        backbone = cfg.backbone.name
        class_list = getattr(cfg.dataset, 'class_list', None)
        if "-" in dataset_name:
            dataset_name = dataset_name.split("-")[0]
        else:
            dataset_name = dataset_name
        dataclass_dict = {
            "cifar10": torchvision.datasets.CIFAR10,
            "cifar100": torchvision.datasets.CIFAR100,
            "utk": UTKDataset,
            "cardio": CardioDataset,
        }
        if dataset_name == "INaturalist":
            dataset = torchvision.datasets.INaturalist(
                root=os.environ["DATAROOT"],
                version="2021_train_mini" if phase == "train" else "2021_valid",
                transform=transform,
                download=True,
            )
        elif dataset_name in dataclass_dict:
            dataset = dataclass_dict[dataset_name](
                root=os.environ["DATAROOT"],
                train=(phase in ["train", "trainval"]),
                download=True,
                transform=transform,
            )
        else:
            raise NotImplementedError
        logger.info(cfg.dataset)
        # create binary dataset
        if hasattr(cfg.dataset, "num_classes"):
            logger.info(f"num_classes: {cfg.dataset.num_classes}")
            dataset.num_classes = cfg.dataset.num_classes
            dataset.classes = torch.arange(cfg.dataset.num_classes)
            # if debug:
                # dataset = torch.utils.data.Subset(dataset, torch.arange(10))
            if class_list is not None:
                logger.info(f"subset indices: {class_list}")
                class_list = torch.tensor(class_list)
                class_indices = torch.tensor([i for i in range(len(dataset)) if dataset.targets[i] in class_list])
                dataset = torch.utils.data.Subset(dataset, class_indices)
                dataset.num_classes = len(class_list)
                # subset.num_classes = len(class_list)
                dataset.subset_indices = class_indices
                # dataset = subset
            elif "binary" in cfg.dataset.name:
                logger.info("binary dataset")
                dataset.classes = torch.tensor([0, 1])
                dataset.num_classes = 2
                dataset.subset_indices = torch.arange(len(dataset))
                #TODO: fix the hard coding
                dataset.targets = [1 if t >= 5 else 0 for t in dataset.targets]
            else:
                print("full dataset", cfg.dataset.name)
                dataset.subset_indices = torch.arange(len(dataset))
            if type(dataset) == torch.utils.data.Subset:
                dataset.targets = [dataset.dataset.targets[i] for i in dataset.indices]
                dataset.targets = torch.tensor(dataset.targets) if not isinstance(dataset.targets, torch.Tensor) else dataset.targets
            else:
                dataset.targets = torch.tensor(dataset.targets) if not isinstance(dataset.targets, torch.Tensor) else dataset.targets
            logger.info(f"dataset: {dataset_name}, phase: {phase}, len: {len(dataset.subset_indices)}")
            logger.info(f"len(dataset): {len(dataset)}")
            class_stats = torch.unique(dataset.targets, return_counts=True)
            logger.info(f"class_stats: {class_stats}")
        return dataset


def split_train_val(trainval_dataset, val_ratio, seed=0):
    # shuffle the indices
    global_seed = torch.initial_seed()
    torch.manual_seed(seed)
    # check if subset
    if type(trainval_dataset) == torch.utils.data.Subset:
        subset_indices = trainval_dataset.indices
    else:
        subset_indices = torch.arange(len(trainval_dataset))
    # shuffle the subset indices
    subset_indices = torch.randperm(len(subset_indices))
    train_ratio = 1 - val_ratio
    train_indices = subset_indices[:int(len(subset_indices) * train_ratio)]
    val_indices = subset_indices[int(len(subset_indices) * train_ratio):]
    torch.manual_seed(global_seed)
    return train_indices, val_indices

class MySubset(torch.utils.data.Dataset):
    #TODO: the performance here is a issue
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        # if hasattr(dataset, "data"):
        #     self.data = [dataset.data[i] for i in indices]
        # else:
        #     self.data = [dataset[i][0] for i in indices]
        # if hasattr(dataset, "targets"):
        #     self.targets = [dataset.targets[i] for i in indices]
        # else:
        #     self.targets = [dataset.targets[i] for i in indices]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

def get_raw_dataset_splits(cfg, transform=None, train_index=None, all_test_transform=False):
    # creat split for train, val, test
    # check if there is a val_ratio
    if hasattr(cfg.dataset, "val_ratio"):
        trainval_dataset = get_raw_dataset(cfg, "train", transform=transform, all_test_transform=all_test_transform)
        test_dataset = get_raw_dataset(cfg, "test", transform=transform, all_test_transform=all_test_transform)
        val_dataset = get_raw_dataset(cfg, "val", transform=transform, all_test_transform=all_test_transform)
        train_indices, val_indices = split_train_val(trainval_dataset, cfg.dataset.val_ratio, seed=cfg.dataset.seed)
        val_dataset = MySubset(trainval_dataset, val_indices)
        if train_index is not None:
            final_train_indices = train_indices[train_index]
        else:
            final_train_indices = train_indices
        train_dataset = MySubset(trainval_dataset, final_train_indices)
        if train_index is not None:
            # logger.debug(f"train_index: {train_index}")
            assert len(train_dataset) == len(train_index)
        return train_dataset, val_dataset, test_dataset
    else:
        train_dataset = get_raw_dataset(cfg, "train", transform=transform, all_test_transform=all_test_transform)
        val_dataset = get_raw_dataset(cfg, "val", transform=transform, all_test_transform=all_test_transform)
        test_dataset = get_raw_dataset(cfg, "test", transform=transform, all_test_transform=all_test_transform)
        return train_dataset, val_dataset, test_dataset

def calculate_eigenvalues_eigenvectors(X):
    n, p = X.shape
    # X is not centered
    X = X
    X_mean = torch.mean(X, dim=0).to(X.device)
    Cov = 1 / n * (X - X_mean).T @ (X - X_mean)
    Cov = Cov.cpu().detach().numpy()
    eigenvalues, eigenvectors = np.linalg.eig(Cov)
    eigenvalues = torch.from_numpy(eigenvalues)
    eigenvectors = torch.from_numpy(eigenvectors)
    return eigenvalues, eigenvectors

def get_output_dir_without_fraction(cfg):
    dataset_dir = f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}" 
    if hasattr(cfg, "pretraining"):
        dataset_dir = f"{dataset_dir}/pretraining#{cfg.pretraining.str}"
    output_dir = f"{dataset_dir}/selection#{cfg.selection.str_without_fraction}"
    return output_dir


def get_dataset_dir(cfg):
    dataset_dir = f"cached_datasets/backbone#{cfg.backbone.str}/dataset#{cfg.dataset.str}" 
    if hasattr(cfg, "pretraining"):
        dataset_dir = f"{dataset_dir}/pretraining#{cfg.pretraining.str}"
    return dataset_dir

def get_output_dir(cfg):
    dataset_dir = get_dataset_dir(cfg)
    output_dir = f"{dataset_dir}/selection#{cfg.selection.str}"
    return output_dir

def get_grad_dir(cfg, phase="train"):
    dataset_dir = get_dataset_dir(cfg)
    grad_dir = f"{dataset_dir}/phase={phase}/grads_full/{cfg.selection.gradient_source}"
    os.makedirs(grad_dir, exist_ok=True)
    return grad_dir

def load_eb_dataset_cfg(cfg=None, device="cpu"):
    if cfg is None:
        with initialize(version_base="1.3.0", config_path="configs"):
            cfg = compose(config_name="default", overrides=["selection=leverage_score"])

    dataset_name = cfg.dataset.name if hasattr(cfg, 'dataset') and hasattr(cfg.dataset, 'name') else "default_dataset_name"
    dataset_dir = get_dataset_dir(cfg)  # Assuming get_dataset_dir is a function that gets the dataset directory from cfg
    # output_dir = get_output_dir(cfg)  # Assuming get_output_dir is a function that gets the output directory from cfg
    class_list = getattr(cfg.dataset, 'class_list', None) if hasattr(cfg, 'dataset') else None

    return load_eb_dataset(
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        class_list=class_list,
        device=device
    )

def load_eb_dataset(dataset_name,
                    dataset_dir,
                    class_list=None,
                    device="cpu") -> dict:
    if dataset_name == "cardio":
        dataset_dict = {}
        for phase in ["train", "val", "test"]:
            dataset = CardioDataset(split=phase)
            X = dataset.data.iloc[:, :-1].values
            Y = dataset.data.iloc[:, -1].values
            dataset_dict[phase] = {
                "X": torch.tensor(X).float().to(device),
                "Y": torch.tensor(Y).float().to(device),
            }
        return dataset_dict

    if "val_ratio=0.0" in dataset_dir:
        phase_list = ["train", "test"]
    else:
        phase_list = ["train", "val", "test"]

    dataset_dict = {}
    for phase in phase_list:
        X_path = f"{dataset_dir}/phase={phase}/"
        print(f"X_path: {X_path}")
        X = torch.load(f"{X_path}X.pt", map_location=device).float()
        Y = torch.load(f"{X_path}Y.pt", map_location=device).float()

        eigenvalues_path = f"{X_path}eigenvalues.pt"
        eigenvectors_path = f"{X_path}eigenvectors.pt"
        if os.path.exists(eigenvalues_path) and os.path.exists(eigenvectors_path):
            eigenvalues = torch.load(eigenvalues_path, map_location=device).float()
            eigenvectors = torch.load(eigenvectors_path, map_location=device).float()
        else:
            eigenvalues, eigenvectors = calculate_eigenvalues_eigenvectors(X)
            torch.save(eigenvalues, eigenvalues_path)
            torch.save(eigenvectors, eigenvectors_path)

        if class_list is not None:
            class_tensor = torch.tensor(class_list).to(device)
            X = X[torch.tensor([y.item() in class_list for y in Y])]
            Y = Y[torch.tensor([y.item() in class_list for y in Y])]
            Y = remap(Y, class_list)

        dataset_dict[phase] = {
            "X": X.to(device),
            "Y": Y.to(device),
            "eigenvalues": eigenvalues.to(device),
            "eigenvectors": eigenvectors.to(device),
        }
    return dataset_dict

def remap(Y, class_list):
    def map_fun(y, class_list):
        return class_list.index(y) if y in class_list else -1

    Y = torch.tensor([map_fun(y.item(), class_list) for y in Y])
    return Y

def calculate_eigenvalues_eigenvectors(X):
    # Placeholder for eigenvalue and eigenvector calculation logic
    eigenvalues = torch.rand(X.shape[1])
    eigenvectors = torch.rand(X.shape[1], X.shape[1])
    return eigenvalues, eigenvectors