#!/bin/fish
python main.py -m --config-name 1_model_pretraining \
backbone=resnet18 \
dataset=cifar10,cifar100 \
hydra/launcher=ds