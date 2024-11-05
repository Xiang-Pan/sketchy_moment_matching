import sys
import json
import logging
from pathlib import Path
from datetime import datetime as dt
import os
import sys
import time
import math
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.init as init


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)



def model_train(epoch,data_loader,net,optimizer, regression=False):
    logging.info('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    score = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs).view(-1)
        total += targets.size(0)
        
        if regression:
            loss = nn.MSELoss()(outputs, targets.view(-1).float())
        else:
            loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        optimizer.step()
        if regression:
            score -= nn.L1Loss(reduction='sum')(outputs, targets.view(-1).float()).item()
        else:
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            score += predicted.eq(targets).sum().item()
    score = score/total
    return score
def model_test(epoch, testloader, net, regression=False):
    
    net.eval()
    test_loss = 0
    score = 0
    total = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs).view(-1)

            
        total += targets.size(0)
        if regression:
            score -= nn.L1Loss(reduction='sum')(outputs, targets.view(-1).float()).item()
        else:
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            
            score += predicted.eq(targets).sum().item()
    score = score/total
    return score



def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class UTKDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.targets = self.df.iloc[:, 1].values
        self.labels = self.targets

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        image = Image.open(img_name)
        age = self.df.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, age
    

class YearBookDataset(torch.utils.data.Dataset):
    def __init__(self, split="train", split_indices_file='yearbook/split_indices.npz', root_dir="yearbook", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.targets = []

        for folder in ['M', 'F']:
            folder_path = os.path.join(root_dir, folder)
            file_names = sorted(os.listdir(folder_path))  # Ensure consistent ordering
            for file_name in file_names:
                if file_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(folder_path, file_name))
                    year = int(file_name.split('_')[0])
                    self.targets.append(year)
        train_indices, test_indices = load_split_indices(split_indices_file)
        if split == "train":
            self.image_paths = [self.image_paths[i] for i in train_indices]
            self.targets = [self.targets[i] for i in train_indices]
        elif split == "test":
            self.image_paths = [self.image_paths[i] for i in test_indices]
            self.targets = [self.targets[i] for i in test_indices]
        self.labels = self.targets
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        target = self.targets[idx] - 1900

        if self.transform:
            image = self.transform(image)

        return image, target
    

def load_split_indices(file_path):
    npzfile = np.load(file_path)
    return npzfile['train_indices'], npzfile['val_indices']