[no-cd]
random:
    python opt.py -m \
    dataset=cardio \
    selection=random \
    selection.fraction=0.001,0.01,0.1,0.2,0.4,0.6,0.8 \
    hydra/launcher=basic

[no-cd]
mlr:
    python opt.py -m \
    dataset=cardio \
    selection=modified_leverage_score \
    selection.fraction=0.001,0.01,0.1,0.2,0.4,0.6,0.8 \
    overwrite=True \
    hydra/launcher=basic \