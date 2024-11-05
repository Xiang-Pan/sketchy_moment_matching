#!/bin/fish

# selection
# python main.py -m \
# mode=selection \
# selection=full \
# hydra/launcher=basic


# python main.py -m \
# mode=linear_probing_selection \
# selection=full \
# hydra/launcher=basic


# python main.py -m \
# mode=selection \
# selection=influence_function \
# training=cifar10_gd \
# hydra/launcher=basic


python main.py -m \
    mode=selection \
    selection=optimization \
    training=cifar10_gd \
    hydra/launcher=basic