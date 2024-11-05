import os
import sys

import argparse
import logging
import pickle
import yaml
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import model
from utils import *
from models import *
from calc_influence_function import calc_s_test_single

def m_guided_opt(S,size):
    n,m = S.shape
    W = cp.Variable(n, boolean=True)
    constaints = [cp.sum(W)==size]
    obj = cp.Minimize(cp.norm(W@S,2))
    prob = cp.Problem(obj, constaints)
    prob.solve(solver=cp.CPLEX)
    W_optim = W.value
 
    return W_optim

def epsilon_guided_opt(S,epsilon):
    n,m = S.shape
    W = cp.Variable(n, boolean=True)
    constaints = [cp.norm(W@S,2)<=epsilon]
    obj = cp.Maximize(cp.sum(W))
    prob = cp.Problem(obj, constaints)
    prob.solve(solver=cp.CPLEX)
    W_optim = W.value
 
    return W_optim

def dataset_pruning(model,all_data,dataset='cifar10'):
    influence_score_list = []
    features,labels = all_data
    for i in tqdm(range(len(features))):
        data,label = features[i].to(device), torch.tensor(labels[i]).to(device)
        influence_score = calc_s_test_single(model,data,label,all_data,gpu=0,recursion_depth=5000, r=3)
        influence_score_list.append(influence_score)
    return influence_score_list



def extract_feature(net, dataloader, dataset='cifar10'):
    feature_list = []
    label_list = []
    net.eval()
    for i, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
        # compute output
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs, _ = net(inputs, out_feature=True)
        feature_list.append(outputs.squeeze().data)
        label_list.append(labels)
    feature_list = torch.cat(feature_list)
    label_list = torch.cat(label_list)
    logging.info(f"feature_list {feature_list.shape}")
    logging.info(f"label_list {label_list.shape}")
    return feature_list, label_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Pruning')
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    
    parser.add_argument('--dataset', default='cifar10',
                        type=str, help='dataset: cifar10/cifar100')
    parser.add_argument('--src_net', default='resnet18', type=str, help='source model')
    parser.add_argument('--tgt_net', default='resnet18', type=str, help='target model')
    
    
    parser.add_argument('--m', default=0, type=int, help='pruning size')
    parser.add_argument('--batchsize', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=200, type=int, help='pretraining epoch')
    
    
    
    args = parser.parse_args()
    args.output_dir = "outputs/optimization/" + str(args).replace(", ", "/").replace("'", "").replace(
        "(", ""
    ).replace(")", "").replace("Namespace", "")
    os.system("rm -rf " + args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        filename=f"{args.output_dir}/train.log",
        level=logging.DEBUG,
        filemode="w",
        datefmt="%H:%M:%S",
        format="%(asctime)s :: %(levelname)-8s \n%(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)    
    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or

    # Data
    logging.info('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batchsize, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batchsize, shuffle=False, num_workers=1)

    wholedataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_test)

    wholeloader = torch.utils.data.DataLoader(
        wholedataset, batch_size=args.batchsize, shuffle=False, num_workers=1)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    logging.info('Cached feature not found, training from scratch..')

    if args.src_net == 'senet':
        net = SeNet()
    elif args.src_net == 'resnet18':
        net = model.ResNet18()
    elif args.src_net == 'resnet50':
        net = model.ResNet50()
    elif args.src_net == 'my_resnet18':
        net = ResNet18()
    elif args.src_net == 'my_resnet50':
        net = ResNet50()           
    else:
        raise ValueError(f'Invalid source model name: {args.src_net}')
    
    logging.info(f"Model: {args.src_net}, Number of parameters: {count_parameters(net)}")
    net = net.to(device)    
    
    net = net.to(device)
    logging.info(net)

    # Pre-train
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    
    ##### STAGE 1 #####
    for epoch in range(0, args.epochs):
        model_train(epoch,trainloader,net,optimizer)
        model_test(epoch,testloader,net)
        scheduler.step()

    # Extract feature
    features, labels = extract_feature(net, wholeloader, dataset=args.dataset)
    features = features.detach().to(device)

    # # calculate covariance feature matrix with pytorch
    torch.save(features, args.output_dir + "/features.pt" )
    args.cached_feature = args.output_dir + "/features.pt"
    
    classifier = net.linear2
    influence_score_list = dataset_pruning(classifier,all_data=[features,labels])
    if args.dataset=='cifar10':
        S = (-1/50000) * influence_score_list

    # Dataset pruning..
    W = m_guided_opt(S,args.m)
    selected_index = (W==0)

    logging.info(f"Selected indices {selected_index}")
    # Pruned dataset constructing..
    if args.dataset == 'cifar10':
        pruned_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        pruned_dataset.data = pruned_dataset.data[selected_index]
        pruned_dataset.targets = list(
            np.array(pruned_dataset.targets)[selected_index])
        pruned_loader = torch.utils.data.DataLoader(
            pruned_dataset, batch_size=args.batchsize, shuffle=True, num_workers=2)
    logging.info(f"Target {len(pruned_dataset)} samples: {Counter(pruned_dataset.targets)}")
    # Pruned dataset training...
    if args.tgt_net == 'resnet18':
        tgt_net = model.ResNet18()
    elif args.tgt_net == 'resnet50':
        tgt_net = model.ResNet50()
    else:
        raise ValueError(f'Invalid target model name: {args.tgt_net}')
    logging.info(f"Model: {args.tgt_net}, Number of parameters: {count_parameters(tgt_net)}")
    
    tgt_net = tgt_net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(tgt_net.parameters(), lr=.01,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200)

    for epoch in range(0, 200):
        model_train(epoch, pruned_loader, tgt_net, optimizer)
        model_test(epoch, testloader, tgt_net)
        scheduler.step()
