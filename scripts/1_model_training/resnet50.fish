#!/bin/fish
python main.py -m --config-name 1_model_pretraining \
backbone=resnet50 \
dataset=cifar10,cifar100 \
hydra/launcher=ds