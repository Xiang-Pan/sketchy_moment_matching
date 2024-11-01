from torchvision import datasets, transforms
from torch import tensor, long

def CIFAR10(data_path):
    channel = 3
    im_size = (224, 224)
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
    dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test

def CIFAR10_CLIP(data_path):
    channel = 3
    im_size = (224, 224)
    num_classes = 10
    import clip
    print("Using CLIP model for CIFAR10.")
    _, preprocess = clip.load("ViT-B/32", "cpu")
    print(preprocess)
    transform = preprocess
    dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=preprocess)
    dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=preprocess)
    normalize = preprocess.transforms[-1]
    mean = normalize.mean
    std = normalize.std
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test