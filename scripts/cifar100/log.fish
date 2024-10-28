#!/bin/fish
# feature extraction
python main.py -m mode=feature_extraction dataset=cifar100 training=cifar10_coreset

# full selection
python main.py -m \
    dataset=cifar100 \
    mode=selection \
    training=cifar10_gd \
    selection=full \
    hydra/launcher=basic

# random selection uniform_unbalanced
python main.py -m \
    dataset=cifar100 \
    mode=selection \
    training=cifar10_gd \
    selection=uniform_unbalanced \
    selection.fraction=0.8,0.6,0.4,0.2,0.1,0.05,0.01,0.005 \
    hydra/launcher=basic


python main.py -m \
    dataset=cifar100 \
    mode=selection \
    training=cifar10_gd \
    selection=full \
    hydra/launcher=basic

#! random selection uniform_balanced
python main.py -m \
    dataset=cifar100 \
    mode=selection \
    training=cifar10_gd \
    selection=uniform_unbalanced \
    selection.fraction=0.8,0.6,0.4,0.2,0.1,0.05,0.01,0.005 \
    hydra/launcher=basic

#! linear probing uniform_unbalanced
python main.py -m \
    dataset=cifar100 \
    mode=linear_probing_selection \
    training=cifar10_gd \
    selection=uniform_unbalanced \
    selection.fraction=0.8,0.6,0.4,0.2,0.1,0.05,0.01,0.005 \
    hydra/launcher=basic

#! linear probing full
python main.py -m \
    dataset=cifar100 \
    mode=linear_probing_selection \
    training=cifar10_gd \
    selection=full \
    hydra/launcher=basic