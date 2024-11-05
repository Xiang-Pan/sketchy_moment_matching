from torchvision import datasets, transforms
from torch import tensor, long
#* There is issue with clip package in certain machines, use transformers instead
import clip
# import open_clip
import torch
import os
import torchvision
import diskcache
import tqdm
import joblib
from joblib import Memory

cachedir = "./cache"
memory = Memory(cachedir, verbose=1)

@memory.cache
def get_targets(dataset):
    targets = []
    for _, y in tqdm.tqdm(dataset, desc="Getting targets"):
        targets.append(y)
    return torch.tensor(targets, dtype=torch.long)


def StanfordCars(data_path, use_clip=False, use_tinynet=False):
    channel = 3
    im_size = (224, 224)
    num_classes = 196
    if use_clip:
        print("Using CLIP model for CIFAR10.")
        _, preprocess = clip.load("ViT-B/32", "cpu")
        # from transformers import CLIPModel, CLIPProcessor
        # preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # transform = preprocess.feature_extractor
        # print(transform)
        transform = preprocess
        dst_train = torchvision.datasets.StanfordCars(
            root=os.environ["DATAROOT"],
            split="train",
            transform=preprocess,
            download=False,
        )
        dst_test = torchvision.datasets.StanfordCars(
            root=os.environ["DATAROOT"],
            split="test",
            transform=preprocess,
            download=False,
        )
        normalize = transform.transforms[-1]
        mean = normalize.mean
        std = normalize.std
        class_names = dst_train.classes

        dst_train.targets = get_targets(dst_train)
        dst_test.targets = get_targets(dst_test)
        return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
    elif use_tinynet:
        raise NotImplementedError("TinyNet is not supported for CIFAR10.")
        import timm
        print("Using TinyNet model for CIFAR10.")
        model = timm.create_model('tinynet_e.in1k', pretrained=True, num_classes=num_classes)
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=False)
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        normalize = transform.transforms[-1]
        mean = normalize.mean
        std = normalize.std
        dst_train.targets = tensor(dst_train.targets, dtype=long)
        dst_test.targets = tensor(dst_test.targets, dtype=long)
        return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
    else:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        from PIL import Image
        transform = transforms.Compose([
            transforms.Resize(int(224 * (8 / 7)), interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        # dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
        # dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        dst_train = torchvision.datasets.StanfordCars(
            root=os.environ["DATAROOT"],
            split="train",
            transform=transform,
            download=False,
        )
        dst_test = torchvision.datasets.StanfordCars(
            root=os.environ["DATAROOT"],
            split="test",
            transform=transform,
            download=False,
        )

        class_names = dst_train.classes
        dst_train.targets = get_targets(dst_train)
        dst_test.targets = get_targets(dst_test)
        return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
