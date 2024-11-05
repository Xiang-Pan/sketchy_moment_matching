#!/bin/fish
python main.py -m --config-name 4_selection-influence_function \
    dataset=cifar10,cifar100 \
    backbone=resnet18,resnet50 \
    hydra/launcher=basic