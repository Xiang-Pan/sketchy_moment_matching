#!/bin/fish
python main.py --config-name selection-linear_probing \
    selection=influence_function \
    selection.fraction=0.05,0.1,0.2,0.3,0.4,0.8,1.0 \
training.rep=null \
training.arch=linear,mlp_50 \
hydra/launcher=ds

python main.py --config-name selection-linear_probing \
    selection=influence_function \
    selection.fraction=0.05,0.1,0.2,0.3,0.4,0.8,1.0 \
training.rep=mlp_50 \
training.arch=linear \
hydra/launcher=ds



python main.py -m --config-name selection-linear_probing \
    selection=uniform_balanced \
    selection.fraction=0.05,0.1,0.2,0.3,0.4,0.6,0.8,1.0 \
training.rep=null \
training.arch=linear \
hydra/launcher=ds

python main.py -m --config-name selection-linear_probing \
    selection=uniform_balanced \
    selection.fraction=0.05,0.1,0.2,0.3,0.4,0.6,0.8,1.0 \
training.rep=mlp_50 \
training.arch=linear \
hydra/launcher=ds

