from torchvision import datasets, transforms
from torch import tensor, long


def CIFAR100(data_path, use_clip, use_tinynet):
    channel = 3
    im_size = (224, 224)
    num_classes = 100
    if use_clip:
        import clip
        print("Using CLIP model for CIFAR10.")
        _, preprocess = clip.load("ViT-B/32", "cpu")
        print(preprocess)
        transform = preprocess
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=preprocess)
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=preprocess)
        normalize = preprocess.transforms[-1]
        mean = normalize.mean
        std = normalize.std
        class_names = dst_train.classes
        dst_train.targets = tensor(dst_train.targets, dtype=long)
        dst_test.targets = tensor(dst_test.targets, dtype=long)
        return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
    elif use_tinynet:
        import timm
        print("Using TinyNet model for CIFAR10.")
        model = timm.create_model('tinynet_e.in1k', pretrained=True, num_classes=num_classes)
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=False)
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        normalize = transform.transforms[-1]
        mean = normalize.mean
        std = normalize.std
        dst_train.targets = tensor(dst_train.targets, dtype=long)
        dst_test.targets = tensor(dst_test.targets, dtype=long)
        return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
    
    
    
    
    
    mean = [0.5071, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2762]
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)
    dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
    class_names = dst_train.classes
    dst_train.targets = tensor(dst_train.targets, dtype=long)
    dst_test.targets = tensor(dst_test.targets, dtype=long)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
