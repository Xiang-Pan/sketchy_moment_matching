# Sketchy Moment Matching (SkMM)
Official implementation of [Sketchy Moment Matching](https://arxiv.org/abs/2407.06120).
<!-- figure -->
<p align="center">
  <img src="./figures/skmm.png" width="500">
</p>

# File Structure
- synthetic/*: synthetic experiments
- hoang/*: the implementation for regression and CIFAR-10-CLIP
- main.py: feature extraction
- data_select.py: do data selection and cache the index
- linear_probe.py: linear probe the model with sklearn
- finetune.py: finetune the model using the cached index (Adam FT-1 and FT-2)

# Environment Variables
- DATAROOT: the root directory for the dataset
- LABROOT: the root directory for the project
- server: the slurm server config, basic if local

# How to run

## Synthetic Experiments
```bash
synthetic/toy.ipynb
```

## Baselines
To run the baselines for StanfordCars
```bash
libs/DeepCore/*.slurm
```

## SkMM Pipeline
```bash
# install task first
conda install conda-forge::go-task
```

```bash
# extract the feature
task feature backbone=clip server=basic

# extract the gradient and sketching
python grad_utils.py -m dataset=cifar10,stanfordcars backbone=resnet18,clip +sketching_dim=16,32,64,128,256,512 +layers=-1,-2 +use_target=random hydra/launcher=basic

# CIFAR10
task cifar10:resnet18:cov:cov1:sweep            # for selection one layer
task cifar10:resnet18:cov:ft1:sweep             # for finetune one layer
task cifar10:resnet18:cov_ntk:selection2:sweep  # for selection two layers
task cifar10:resnet18:cov_ntk:ft2:sweep         # for finetune two layers

# StanfordCars
task stanfordcars:clip:cov:cov1:sweep                   # for selection one layer
task stanfordcars:clip:cov:ft1:sweep                    # for finetune one layer
task stanfordcars:resnet18:cov_ntk:selection2:sweep     # for selection two layers
task stanfordcars:resnet18:cov_ntk:ft2:sweep            # for finetune two layers
```

```bash
# To get the summary table, change the table to other configs you would like to have
python notebooks/export.py +table=StanfordCarsResNet18FT1 
```