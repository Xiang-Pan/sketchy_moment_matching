from torchvision import datasets, transforms
from torch import tensor, long
import clip


def CIFAR100(data_path):
    channel = 3
    im_size = (224, 224)
    num_classes = 100
    mean = [0.5071, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2762]
    transform = transforms.Compose([transforms.Resize(im_size), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)
    dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test

def CIFAR100_CLIP(data_path):
    _, preprocess = clip.load('ViT-B/32', "cpu")
    im_size = (224, 224)
    channel = 3
    num_classes = 100
    mean = [0.5071, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2762]
    dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=preprocess)
    dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=preprocess)
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test