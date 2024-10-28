import torch
from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import Dataset


class UTKDataset(Dataset):
    def __init__(self, root, train, transform=None, download=False):
        super(UTKDataset, self).__init__()
        dataroot = f"{root}/utkface_aligned_cropped/UTKFace"
        # [age]_[gender]_[race]_[date&time].jpg.chip.jpg
        self.files = os.listdir(dataroot)
        self.files = [f for f in self.files if f.endswith('.jpg')]
        # I only interested in age
        age_range = [int(f.split('_')[0]) for f in self.files]
        self.age_range = (min(age_range), max(age_range))
        # remap it to [0, 1]
        self.age_norm = lambda x: (x - self.age_range[0]) / (self.age_range[1] - self.age_range[0])
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        if train:
            self.files = self.files[:int(0.9 * len(self.files))]
        else:
            self.files = self.files[int(0.9 * len(self.files)):]
        if train:
            self.transform = self.train_transform
        else:
            self.transform = self.test_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        data_root = os.getenv('DATA_ROOT')
        img = Image.open(f'{data_root}/utkface_aligned_cropped/UTKFace/{f}')
        if self.transform:
            img = self.transform(img)
        age = int(f.split('_')[0])
        age = self.age_norm(age)
        return img, age


if __name__ == "__main__":
    dataset = UTKDataset()
