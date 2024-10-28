#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100GB
#SBATCH --job-name=prune
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate /scratch/hvp2011/envs/python
nvidia-smi

cd /scratch/hvp2011/implement/code_datasetptuning/

m=1000
c=1.0
dataset=svhn
dataset=fmnist
dataset=cifar
# for m in 1000 2000 3000 4000
# do
# python main.py --lr .001 --m $m --reg_tradeoff 1 --pgd_lr .01  --c $c  --seed 0 --dataset $dataset --retrain_lr .01 --random_prune False --cached_feature "outputs/2nd_phase/seed=1/lr=0.001/retrain_lr=0.01/pgd_lr=0.01/dataset=cifar/cached_feature=None/m=1000/batchsize=128/epochs=200/random_prune=False/c=1.0/reg_tradeoff=1.0/alpha=2/features.pt"
# python main.py --lr .001 --m $m --reg_tradeoff 1 --pgd_lr .01  --c $c  --seed 1 --dataset $dataset --retrain_lr .01 --random_prune False --cached_feature "outputs/2nd_phase/seed=1/lr=0.001/retrain_lr=0.01/pgd_lr=0.01/dataset=cifar/cached_feature=None/m=1000/batchsize=128/epochs=200/random_prune=False/c=1.0/reg_tradeoff=1.0/alpha=2/features.pt"
# python main.py --lr .001 --m $m --reg_tradeoff 1 --pgd_lr .01  --c $c  --seed 2 --dataset $dataset --retrain_lr .01 --random_prune False --cached_feature "outputs/2nd_phase/seed=1/lr=0.001/retrain_lr=0.01/pgd_lr=0.01/dataset=cifar/cached_feature=None/m=1000/batchsize=128/epochs=200/random_prune=False/c=1.0/reg_tradeoff=1.0/alpha=2/features.pt"

# done
# seed=0
# for batchsize in 32 64 128
# do 
#     for m in 1000 2000 3000 4000
#     do
#         for c in 1 1.05
#         do
#         python main.py --lr .001 --m $m --reg_tradeoff 1 --pgd_lr .01  --c $c  --seed $seed --dataset $dataset --retrain_lr .001 --batchsize $batchsize --random_prune False --cached_feature outputs/2nd_phase/seed=0/lr=0.001/retrain_lr=0.001/pgd_lr=0.01/dataset=svhn/cached_feature=None/m=1000/batchsize=128/epochs=200/random_prune=False/c=5.0/reg_tradeoff=1.0/alpha=2/features.pt
#         done
#     done
# done

# seed=2
# for batchsize in 32
# do 
#     for m in 1000 2000 3000 4000
#     do
#         for c in 1.
#         do
#         python main.py --lr .001 --m $m --reg_tradeoff 1 --pgd_lr .01  --c $c  --seed $seed --dataset $dataset --retrain_lr .001 --batchsize $batchsize --random_prune False --cached_feature outputs/2nd_phase/seed=0/lr=0.001/retrain_lr=0.01/pgd_lr=0.01/dataset=fmnist/cached_feature=None/m=1000/batchsize=128/epochs=200/random_prune=False/c=5.0/reg_tradeoff=1.0/alpha=2/features.pt
#         done
#     done
# done
# m=1000
# # dataset=svhn
# dataset=fmnist

# for seed in 0 1 2
# do
#     for batchsize in 32 64 128 256
#     do
#         # python main.py --lr .001 --m $m --reg_tradeoff 1 --pgd_lr .01  --c 1  --seed $seed --dataset $dataset --retrain_lr .001 --batchsize $batchsize --random_prune False 
#         for m in 1000 2000 3000 4000
#         do
#             python main.py --lr .001 --m $m --reg_tradeoff 1 --pgd_lr .01  --c 1  --seed $seed --dataset $dataset --retrain_lr .001 --batchsize $batchsize --random_prune True
#         done
#     done
# done

dataset=cifar100
# python main.py --epochs 0 --lr .001 --m $m --reg_tradeoff $reg_tradeoff --pgd_lr .01  --c 1.0  --seed $seed --dataset $dataset --retrain_lr .003 --random_prune False --batchsize 128 



for seed in 0 1 2
do
    for reg_tradeoff in  .1
    do
        # python main.py --epochs 0 --lr .001 --m 4000 --reg_tradeoff $reg_tradeoff --pgd_lr .01  --c 1.0  --seed $seed --dataset $dataset --retrain_lr .01 --random_prune False --batchsize 128 
        for m in 100 1000 2000 3000 4000
        do
            # python main.py --lr .001 --m $m --reg_tradeoff $reg_tradeoff --pgd_lr .01  --c 1  --seed 0 --dataset cifar --retrain_lr .001 --random_prune False --cached_feature "outputs/2nd_phase/seed=1/lr=0.001/retrain_lr=0.01/pgd_lr=0.01/dataset=cifar/cached_feature=None/m=1000/batchsize=128/epochs=200/random_prune=False/c=1.0/reg_tradeoff=1.0/alpha=2/features.pt"
            # python main.py --lr .001 --m $m --reg_tradeoff $reg_tradeoff --pgd_lr .01  --c 1  --seed 0 --dataset svhn --retrain_lr .01 --random_prune False --cached_feature "outputs/2nd_phase/seed=1/lr=0.001/retrain_lr=0.001/pgd_lr=0.01/dataset=svhn/cached_feature=None/m=1000/batchsize=256/epochs=200/random_prune=False/c=1.0/reg_tradeoff=1.0/alpha=2/features.pt"
            # python main.py --epochs 0 --lr .001 --m $m --reg_tradeoff $reg_tradeoff --pgd_lr .01  --c 1.0  --seed $seed --dataset cifar100 --retrain_lr .1 --random_prune False --batchsize 128 
            # python main.py --epochs 0 --lr .001 --m $m --reg_tradeoff $reg_tradeoff --pgd_lr .01  --c 1.8  --seed $seed --dataset cifar10 --retrain_lr .01 --batchsize 128 
            # python main.py --epochs 0 --lr .001 --m $m --reg_tradeoff $reg_tradeoff --pgd_lr .01  --c 1.0  --seed $seed --dataset utk --retrain_lr .1 --batchsize 128 
            python main.py --epochs 0 --lr .001 --m $m --reg_tradeoff $reg_tradeoff --pgd_lr .01  --c 1.4  --random_prune False --seed $seed --dataset yearbook --retrain_lr 1 --batchsize 128
            
            # python main.py --epochs 0 --lr .001 --m $m --reg_tradeoff $reg_tradeoff --pgd_lr .01  --c 1.0  --seed $seed --dataset cifar100 --retrain_lr .003 --random_prune False --batchsize 128 
            # python main.py --epochs 0 --lr .001 --m $m --reg_tradeoff $reg_tradeoff --pgd_lr .01  --c 1.0  --seed $seed --dataset cifar10_swav --retrain_lr .003 --random_prune False --batchsize 128 
            # python main.py --epochs 0 --lr .001 --m $m --reg_tradeoff $reg_tradeoff --pgd_lr .01  --c 1.0  --seed $seed --dataset cifar100_swav --retrain_lr .0003 --random_prune False --batchsize 128 
            # python main.py --lr .001 --m $m --reg_tradeoff $reg_tradeoff --pgd_lr .01  --c 1  --seed $seed --dataset cifar --retrain_lr .001 --random_prune False --cached_feature "outputs/2nd_phase/seed=1/lr=0.001/retrain_lr=0.01/pgd_lr=0.01/dataset=cifar/cached_feature=None/m=1000/batchsize=128/epochs=200/random_prune=False/c=1.0/reg_tradeoff=1.0/alpha=2/features.pt"
            # python main.py --lr .001 --m $m --reg_tradeoff $reg_tradeoff --pgd_lr .01  --c 1  --seed $seed --dataset svhn --retrain_lr .001 --random_prune False --cached_feature "outputs/2nd_phase/seed=1/lr=0.001/retrain_lr=0.001/pgd_lr=0.01/dataset=svhn/cached_feature=None/m=1000/batchsize=256/epochs=200/random_prune=False/c=1.0/reg_tradeoff=1.0/alpha=2/features.pt"
            # python main.py --lr .001 --m $m --reg_tradeoff $reg_tradeoff --pgd_lr .01  --c 1  --seed $seed --dataset fmnist --retrain_lr .001 --random_prune False --cached_feature "outputs/2nd_phase/seed=2/lr=0.001/retrain_lr=0.001/pgd_lr=0.01/dataset=fmnist/cached_feature=None/m=1000/batchsize=32/epochs=200/random_prune=False/c=1.0/reg_tradeoff=1.0/alpha=2/features.pt"

        done
    done 
done


# python main.py --epochs 0 --lr .001 --m 1000 --reg_tradeoff 1 --pgd_lr .01  --c 1  --seed 0 --dataset cifar100 --retrain_lr .1 --random_prune True --batchsize 128 
# python main.py --epochs 0 --lr .001 --m 2000 --reg_tradeoff 1 --pgd_lr .01  --c 1  --seed 0 --dataset cifar10 --retrain_lr .1 --random_prune True --batchsize 128 
# python main.py --epochs 0 --lr .001 --m 3000 --reg_tradeoff 1 --pgd_lr .01  --c 1  --seed 0 --dataset cifar10 --retrain_lr .1 --random_prune True --batchsize 128 
# python main.py --epochs 0 --lr .001 --m 4000 --reg_tradeoff 1 --pgd_lr .01  --c 1  --seed 0 --dataset cifar10 --retrain_lr .1 --random_prune True --batchsize 128 

# python main.py --epochs 0 --lr .001 --m 1000 --reg_tradeoff 1 --pgd_lr .01  --c 1  --seed 1 --dataset cifar10 --retrain_lr .1 --random_prune True --batchsize 128 
# python main.py --epochs 0 --lr .001 --m 2000 --reg_tradeoff 1 --pgd_lr .01  --c 1  --seed 1 --dataset cifar10 --retrain_lr .1 --random_prune True --batchsize 128 
# python main.py --epochs 0 --lr .001 --m 3000 --reg_tradeoff 1 --pgd_lr .01  --c 1  --seed 1 --dataset cifar10 --retrain_lr .1 --random_prune True --batchsize 128 
# python main.py --epochs 0 --lr .001 --m 4000 --reg_tradeoff 1 --pgd_lr .01  --c 1  --seed 1 --dataset cifar10 --retrain_lr .1 --random_prune True --batchsize 128 

# python main.py --epochs 0 --lr .001 --m 1000 --reg_tradeoff 1 --pgd_lr .01  --c 1  --seed 2 --dataset cifar10 --retrain_lr .1 --random_prune True --batchsize 128 
# python main.py --epochs 0 --lr .001 --m 2000 --reg_tradeoff 1 --pgd_lr .01  --c 1  --seed 2 --dataset cifar10 --retrain_lr .1 --random_prune True --batchsize 128 
# python main.py --epochs 0 --lr .001 --m 3000 --reg_tradeoff 1 --pgd_lr .01  --c 1  --seed 2 --dataset cifar10 --retrain_lr .1 --random_prune True --batchsize 128 
# python main.py --epochs 0 --lr .001 --m 4000 --reg_tradeoff 1 --pgd_lr .01  --c 1  --seed 2 --dataset cifar10 --retrain_lr .1 --random_prune True --batchsize 128 

# python main.py --lr .001 --m 1000 --reg_tradeoff 0 --pgd_lr .01  --c 1  --seed 0 --dataset svhn --retrain_lr .001 --random_prune True --cached_feature "outputs/2nd_phase/seed=1/lr=0.001/retrain_lr=0.001/pgd_lr=0.01/dataset=svhn/cached_feature=None/m=1000/batchsize=256/epochs=200/random_prune=False/c=1.0/reg_tradeoff=1.0/alpha=2/features.pt"


# python main.py --epochs 0 --lr .001 --m 1000 --reg_tradeoff 0.003 --pgd_lr .001  --c 1.  --seed $seed --dataset $dataset --retrain_lr .01 --random_prune False --batchsize 128 

# python main.py --lr .001 --m $m --reg_tradeoff 1 --pgd_lr .01  --c $c  --seed 0 --dataset $dataset --retrain_lr .001 --random_prune False --batchsize 256 

# python main.py --lr .001 --m $m --reg_tradeoff 1 --pgd_lr .01  --c $c  --seed 2 --dataset $dataset --retrain_lr .001 --random_prune True --batchsize 32


