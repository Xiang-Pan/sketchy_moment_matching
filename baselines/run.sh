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



for seed in 0 1 2
do
    for reg_tradeoff in   .001
    do
        for m in 1000 2000 3000 4000
        do
            python main.py --epochs 0 --lr .001 --m $m --reg_tradeoff $reg_tradeoff --pgd_lr .01  --c .95  --seed $seed --dataset cifar10 --retrain_lr .01 --random_prune False --batchsize 128 
            

        done
    done 
done




