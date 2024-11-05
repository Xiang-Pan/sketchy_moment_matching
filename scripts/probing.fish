#!/bin/fish
# selection and probing
set fraction 2e-4,3e-1,2e-1,1e-1,1e0
set seed 0,1,2,3,4

python main.py -m mode=selection selection.fraction=$fraction selection.seed=$seed hydra/launcher=ai
python main.py -m mode=linear_probing_selection selection.fraction=$fraction selection.seed=$seed hydra/launcher=ai