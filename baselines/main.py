import os
import sys

import argparse
import logging
import pickle
import yaml
from tqdm import tqdm
from collections import Counter

import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary

import clip
import model
from utils import *
from models import *


def extract_feature(net, dataloader, dataset='cifar10'):
    feature_list = []
    label_list = []
    net.eval()
    with torch.no_grad():
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
    
    # convert to torch float32
    # feature_list = feature_list.float()

    return feature_list, label_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Pruning')
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--retrain_lr', default=0.01, type=float, help='retraining learning rate')
    parser.add_argument('--pgd_lr', default=0.1, type=float, help='projected gradient descent learning rate')
    
    parser.add_argument('--dataset', default='cifar10',
                        type=str, help='dataset: cifar10/cifar100')
    parser.add_argument('--cached_feature', default=None, type=str, help='cached feature path') 
    
    parser.add_argument('--m', default=0, type=int, help='pruning size')
    parser.add_argument('--batchsize', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=200, type=int, help='pretraining epoch')
    
    parser.add_argument('--c', default=10, type=float, help='C')
    parser.add_argument('--reg_tradeoff', default=0.1, type=float, help='reg')
    parser.add_argument('--alpha', default=2, type=float, help='alpha')
    
    args = parser.parse_args()
    args.output_dir = "outputs/2nd_phase/" + str(args).replace(", ", "/").replace("'", "").replace(
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
    logging.info(f"Parameters: {args}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    # Data
    logging.info('==> Preparing data..')
    tgt_scheduler = None
    
    if args.dataset == 'cifar':
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

        net = CNNForCIFAR10()
        tgt_net = model.ResNet18()
        

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

        wholedataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_test)
        
        pruned_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        
        tgt_optimizer = optim.SGD(tgt_net.parameters(), lr=args.retrain_lr,
                            momentum=0.9, weight_decay=5e-4)
        tgt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            tgt_optimizer, T_max=200)  
        
    elif args.dataset == 'cifar10':
       
        _, preprocess = clip.load('ViT-B/32', device)
        
        net = LinearCLIP('ViT-B/32', num_classes=10)
        tgt_net = LinearCLIP('ViT-B/32', num_classes=10)
        
        transform_train = preprocess
        transform_test = preprocess

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

        wholedataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_test)
        
        pruned_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        
        # tgt_optimizer = optim.SGD(tgt_net.parameters(), lr=args.retrain_lr,
        #                     momentum=0.9, weight_decay=5e-4)
        # tgt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     tgt_optimizer, T_max=200)  
        
        tgt_optimizer = optim.Adam(tgt_net.parameters(), lr=args.retrain_lr)
        tgt_scheduler = None 
    elif args.dataset == 'cifar':
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

        net = CNNForCIFAR10()
        tgt_net = model.ResNet18()
        

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

        wholedataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_test)
        
        pruned_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        
        tgt_optimizer = optim.SGD(tgt_net.parameters(), lr=args.retrain_lr,
                            momentum=0.9, weight_decay=5e-4)
        tgt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            tgt_optimizer, T_max=200)  
        
    elif args.dataset == 'cifar10_swav':
       
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
        )
        ])

        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
                )
        ])

        net = SwAVResNet50(num_classes=10)
        tgt_net = SwAVResNet50(num_classes=10)
        

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)

        wholedataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_test)
        
        pruned_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        
        tgt_optimizer = optim.SGD(tgt_net.parameters(), lr=args.retrain_lr,
                            momentum=0.9, weight_decay=5e-4)
        tgt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            tgt_optimizer, T_max=200)       
         
    elif args.dataset == 'cifar100':
        
        _, preprocess = clip.load('ViT-B/32', device)
        
        net = LinearCLIP('ViT-B/32', num_classes=100)
        tgt_net = LinearCLIP('ViT-B/32', num_classes=100)
        
        transform_train = preprocess
        transform_test = preprocess

        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)

        wholedataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_test)
        
        pruned_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        
        tgt_optimizer = optim.SGD(tgt_net.parameters(), lr=args.retrain_lr,
                            momentum=0.9, weight_decay=5e-4)
        tgt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            tgt_optimizer, T_max=200)  
        
        tgt_optimizer = optim.Adam(tgt_net.parameters(), lr=args.retrain_lr)
        tgt_scheduler = None  

    elif args.dataset == 'cifar100_swav':
        
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
        )
        ])

        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
                )
        ])
        
        net = SwAVResNet50(num_classes=100)
        tgt_net = SwAVResNet50(num_classes=100)

        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)

        wholedataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_test)
        
        pruned_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        
        tgt_optimizer = optim.SGD(tgt_net.parameters(), lr=args.retrain_lr,
                            momentum=0.9, weight_decay=5e-4)
        tgt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            tgt_optimizer, T_max=200)  
        
        # tgt_optimizer = optim.Adam(tgt_net.parameters(), lr=args.retrain_lr)
        # tgt_scheduler = None  
            
    elif args.dataset == 'tiny_imagenet':  
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

        # net = CNNForCIFAR10()
        # tgt_net = model.ResNet18()
        
        # net = SwAVResNet50()
        # tgt_net = SwAVResNet50()
        
        _, preprocess = clip.load('ViT-B/32', device)
        
        net = LinearCLIP('ViT-B/32', 200)
        tgt_net = LinearCLIP('ViT-B/32', 200)
        
        transform_train = preprocess
        transform_test = preprocess
        
        # trainset = torchvision.datasets.ImageFolder(root=os.path.join('./data', 'tiny-imagenet-200/train'), transform=transform)
        # dst_test = torchvision.datasets.ImageFolder(root=os.path.join('./data', 'tiny-imagenet-200/test'), transform=transform)
        
        trainset = torchvision.datasets.ImageFolder(root=os.path.join('./data', 'tiny-imagenet-200/train'),  transform=transform_train)

        testset = torchvision.datasets.ImageFolder(root=os.path.join('./data', 'tiny-imagenet-200/test'), transform=transform_test)

        wholedataset = torchvision.datasets.ImageFolder(root=os.path.join('./data', 'tiny-imagenet-200/train'), transform=transform_test)
        
        pruned_dataset = torchvision.datasets.ImageFolder(root=os.path.join('./data', 'tiny-imagenet-200/train'), transform=transform_train)
        
        tgt_optimizer = optim.SGD(tgt_net.parameters(), lr=args.retrain_lr,
                            momentum=0.9, weight_decay=5e-4)
        tgt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            tgt_optimizer, T_max=200)  
        
        tgt_optimizer = optim.Adam(tgt_net.parameters(), lr=args.retrain_lr)
        tgt_scheduler = None          
          

    
    elif args.dataset == 'fmnist':
        
        transform_train = transforms.Compose(
            [ transforms.Resize((32,32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        
        
        transform_test = transforms.Compose(
            [transforms.Resize((32,32)),
                transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        # Create datasets for training & validation, download if necessary
        trainset = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform_train, download=True)
        testset = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform_test, download=True) 
        wholedataset = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform_test, download=True)
        pruned_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform_train)
        net = LeNet()
        tgt_net = LeNet()

        tgt_optimizer = optim.Adam(tgt_net.parameters(), lr=args.retrain_lr)
       
    elif args.dataset == 'svhn':
        transform_train = transforms.Compose([
            transforms.Resize((36,36)),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
        ])

        trainset = torchvision.datasets.SVHN(
            root='./data', split="train", download=True, transform=transform_train)


        testset = torchvision.datasets.SVHN(
            root='./data', split="test", download=True, transform=transform_test)

        wholedataset = torchvision.datasets.SVHN(
            root='./data', split="train", download=True, transform=transform_test)           
        pruned_dataset = torchvision.datasets.SVHN(
            root='./data', split="train", download=True, transform=transform_test)  
        # setattr(pruned_dataset, 'targets', pruned_dataset.labels)
        # setattr(wholedataset, 'targets', wholedataset.labels)
        
        net = CNNForSVHNInner()
        tgt_net = CNNForSVHN()
        tgt_optimizer = optim.Adam(tgt_net.parameters(), lr=args.retrain_lr)   
    
        
    else:
        raise ValueError(f'Invalid dataset name: {args.dataset}')     
    # breakpoint()
    print(f"Trainset {len(trainset)}")
    print(f"Testset {len(testset)}")   
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batchsize, shuffle=False, num_workers=2)
    wholeloader = torch.utils.data.DataLoader(
        wholedataset, batch_size=args.batchsize, shuffle=False, num_workers=2)

    logging.info(f"Source net {summary(net)}")
    
    logging.info(f"Target net {summary(tgt_net)}")

    net = net.to(device)    
    tgt_net = tgt_net.to(device)

    if not args.cached_feature:
        # Model
        logging.info('Cached feature not found, training from scratch..')

        # if args.src_net == 'senet':
        #     net = SeNet()
        # elif args.src_net == 'resnet18':
        #     net = model.ResNet18()
        # elif args.src_net == 'resnet50':
        #     net = model.ResNet50()
        # elif args.src_net == 'my_resnet18':
        #     net = ResNet18()
        # elif args.src_net == 'my_resnet50':
        #     net = ResNet50()           
        # else:
        #     raise ValueError(f'Invalid source model name: {args.src_net}')
        
        logging.info(f"Number of source net parameters: {count_parameters(net)}")
        logging.info(f"Number of target net parameters: {count_parameters(tgt_net)}")
        

        

        # Pre-train
        optimizer = optim.Adam(net.parameters(), lr=args.lr)

        
        ##### STAGE 1 #####
        for epoch in range(0, args.epochs):
            model_train(epoch,trainloader,net,optimizer)
            model_test(epoch,testloader,net)
            # scheduler.step()

        # Extract feature
        features, labels = extract_feature(net, wholeloader, dataset=args.dataset)
        features = features.detach().to(device)
        labels = labels.cpu().numpy()

        # # calculate covariance feature matrix with pytorch
        cov = torch.mm(features, features.t())
        torch.save(features, args.output_dir + "/features.pt" )
        args.cached_feature = args.output_dir + "/features.pt"
    
    ##### STAGE 2 #####
    features = torch.load(args.cached_feature).to(device) 
    
    if args.dataset == 'svhn':
        labels = wholedataset.labels
    else:
        labels = wholedataset.targets
    
    # X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features.cpu().numpy())
    
    # sns.scatterplot(
    #     x=X_tsne[:, 0],
    #     y=X_tsne[:, 1],
    #     hue=labels,
    # )        
    
    # plt.savefig(args.output_dir + "/tsne.png")
    # plt.close()
    
    cov = features.t()@features/len(features)
    logging.info(f"covariance {cov.shape}")
    U, S, V =  torch.svd(cov)
    logging.info(f"U {U.shape}")
    logging.info(f"S {S.shape}")
    logging.info(f"V {V.shape}")
    logging.info(f"S {S}")
    logging.info(f"U-V {U-V}")
    # logging.info("U.t()@U", U.t()@U)
    # logging.info("V@V.t()", V@V.t())
    # new_S = torch.diag_embed(S)
    # logging.info("S", S)
    # logging.info("new_S", new_S)
    # logging.info("Error", (features - U@torch.diag_embed(S)@V.t()))
    # logging.info("Error 2", cov - V@new_S@U.t()@U@new_S@V.t())
    # logging.info("Error 3", cov - V@(new_S**2)@V.t(), torch.norm(cov - V@(new_S**2)@V.t()))
    logging.info(f"Error {cov - U@torch.diag_embed(S)@U.t(), torch.norm(cov - U@torch.diag_embed(S)@U.t())}")
    logging.info(f"Covariance norm {torch.norm(cov)}")    
    
    
    
    gamma = (torch.randn(cov.shape[0]).to(device)+1)/2
    gamma = gamma.requires_grad_(True)
    

    w = nn.Parameter(torch.randn((len(features),1)).to(device) , requires_grad=True)
    w = torch.nn.Parameter(torch.zeros((len(features),1)).to(device), requires_grad=True)
    U = U.detach()
    S = S.detach()
    features = features.detach()
    
    logging.info(f"Loss {torch.norm(S**2 - torch.diag(U.t()@cov@U))}")
    optimizer = optim.Adam([w], lr=args.pgd_lr)
    mse_losses = []
    reg_losses = []
    total_losses = []
    
    
    
    for iter in range(20000):
        W = torch.softmax(w, dim=0)
        # gamma = torch.clamp(gamma, 1/args.c, args.c).detach().requires_grad_(True)
        gamma = torch.clamp(gamma, args.c).detach().requires_grad_(True)
        
        
        # # inspect #   
        # selected_index = torch.topk(torch.randn(len(trainset)) , args.m).indices
        # W = [1 if i in selected_index else 0 for i in range(len(W))]
        # W = torch.tensor(W, dtype=torch.float).cuda()
        # W = torch.softmax(W, -1).reshape(-1,1)
        # new_features = W*features
        # new_cov = new_features.t()@features
        # print(torch.sort(torch.diag(U.t()@new_cov@U)/S))
        # # inspect #
        
        new_features = W*features
        # new_cov = new_features.t()@new_features/len(features)
        # mse = torch.norm(cov - new_cov)
        new_cov = new_features.t()@features
        mse = torch.norm(S*gamma - torch.diag(U.t()@new_cov@U))
        
        
        # reg = ((W**args.alpha)*((1/args.m-W)**args.alpha)).sum()
        reg = 1/torch.norm(W)
        loss =  mse + args.reg_tradeoff * reg
        
        loss.backward()
        
        lr = 1e-2*args.pgd_lr/(iter+1)
        
        new_gamma = (gamma - lr * gamma.grad).detach().clone().requires_grad_(True)
        optimizer.step()
        
        w.grad.zero_()
        gamma.grad.zero_()
        gamma = new_gamma 
        
        if iter%500 == 0:
            logging.info(f"iter {iter}, loss {loss.item()}, mse {mse.item()}, reg {reg.item()}")
        mse_losses.append(mse.item())
        reg_losses.append(reg.item())
        total_losses.append(loss.item())
    
    plt.plot(mse_losses)
    plt.savefig(args.output_dir + "/mse_losses.png")
    plt.title("MSE Loss")
    plt.close()
    plt.plot(reg_losses)
    plt.title("Regularization Loss")
    plt.savefig(args.output_dir + "/reg_losses.png")
    plt.close()
    plt.plot(total_losses)
    plt.title("Total Loss")
    plt.savefig(args.output_dir + "/total_losses.png")
    plt.close()
    
    # Create a histogram
    plt.hist(W.detach().cpu().numpy(), bins=300, color='blue', edgecolor='black')

    # Add titles and labels
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig(args.output_dir + "/hist.png")
    plt.close()
    # Baseline

    W = W.reshape(-1)

    if args.dataset == 'fmnist':
        labels = labels.cpu().numpy()
    counter = Counter(labels)
    selected_index = torch.tensor([], dtype=torch.long).to(device)
    
    labels = torch.tensor(labels).to(device)
    logging.info(f"Whole dataset: {len(wholedataset)} samples: {counter}")
    
    if "100" in args.dataset:
        icp = int(args.m/100)
    else:
        icp = int(args.m/10)
    
    
    for i, (class_idx, n_samples) in enumerate(counter.items()):
        w_class = W*(labels==class_idx)
        # selected_index = torch.cat([selected_index, torch.topk(w_class , icp).indices])
        n_select = int(n_samples/len(trainset)*args.m)
        if i == len(counter)-1:
            n_select = args.m - len(selected_index)
                
        # selected_index = torch.cat([selected_index, torch.topk(w_class , icp).indices])
        selected_index = torch.cat([selected_index, torch.topk(w_class , n_select).indices])
    selected_index = selected_index.cpu().numpy()
    logging.info(f"Selected indices {selected_index}")
    # Pruned dataset constructing..



    # pruned_dataset.data = pruned_dataset.data[selected_index]
    pruned_dataset = torch.utils.data.Subset(pruned_dataset, selected_index)
    if args.dataset == 'svhn':
        # labels = list(
        #     np.array(pruned_dataset.labels)[selected_index])
        # pruned_dataset.labels = labels
        labels = list(np.array(pruned_dataset.dataset.labels)[selected_index])
    else:
        # labels = list(
        #     np.array(pruned_dataset.targets)[selected_index])
        # pruned_dataset.targets = labels
        labels = list(np.array(pruned_dataset.dataset.targets)[selected_index])
        
    pruned_loader = torch.utils.data.DataLoader(
        pruned_dataset, batch_size=args.batchsize, shuffle=True, num_workers=2)
    try:
        logging.info(f"Target {len(pruned_dataset)} samples: {Counter(labels.cpu().numpy())}")
    except:
        logging.info(f"Target {len(pruned_dataset)} samples: {Counter(labels)}")
        
    
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(0, 200):
        train_acc = model_train(epoch, pruned_loader, tgt_net, tgt_optimizer)
        test_acc = model_test(epoch, testloader, tgt_net)
        logging.info(f"Epoch {epoch}, Train Acc {train_acc}, Test Acc {test_acc}")
        if test_acc > best_acc:
            best_acc = test_acc
            logging.info(f"Best model found at epoch {epoch}, acc {best_acc}!")
        if tgt_scheduler is not None:
            tgt_scheduler.step()
