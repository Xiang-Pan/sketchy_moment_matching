#!/bin/fish
# do selection
# python main.py -m \
#     mode=selection \
#     selection.fraction=0.1,0.2,0.4,0.6,0.8,1.0 \
#     selection.seed=0 \
#     hydra/launcher=basic

# test selection
python main.py -m \
    mode=linear_probing_selection \
    selection.fraction=1.0 \
    selection.seed=0 \
    training.train_batch_size=50000 \
    training.optimizer.lr=0.01 \
    hydra/launcher=basic
