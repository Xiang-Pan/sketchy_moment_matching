#!/bin/fish
# get the cached_datasets/backbone=resnet50-swav_run0/dataset=cifar10-0.01-0/selection=cov-eb-0.0002-0-2
# get folders under cached_datasets/backbone=resnet50-swav_run0/dataset=cifar10-0.01-0/selection=cov-eb-0.0002-0-2
set folders (find cached_datasets/backbone=resnet50-swav_run0/dataset=cifar10-0.01-0/selection=cov-eb-0.0002-0-2 -type d)
# split the folder name by / get the last layer
set ratio_list ""
for folder in $folders
    set folder_name (string split / $folder)
    set folder_name $folder_name[-1]
    set ratio (string split = $folder_name)[-1]
    echo $ratio
    set ratio_list "$ratio_list,$ratio"
end
echo $ratio_list

# get the last layer of the folder name
# python main.py -m mode=linear_probing_selection selection=cov selection.c=10 selection.fraction=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 hydra/launcher=ai