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

cd /scratch/hvp2011/implement/code_datasetptuning/DeepCore


# method=Uniform
# method=ContextualDiversity
# method=Glister
# method=kCenterGreedy
# method=Herding
# method=GraNd
# method=GradMatch
# method=Forgetting
method=DeepFool
# method=Craig
# method=Uncertainty
python -u main.py --seed 0  --fraction 0.02 --dataset CIFAR10 --data_path ./data --num_exp 1 --workers 4 --optimizer Adam -se 0 --selection Uniform --model LinearCLIP --lr 0.01 --weight_decay 0 -sp ./result --batch 128

for seed in  0 1 2 
do
    for fraction in .02 .04 .06 .08
    do 
    # --uncertainty Margin LeastConfidence 
    python -u main.py --seed $seed  --fraction $fraction --dataset CIFAR10 --data_path ../data --num_exp 1 --workers 4 --optimizer Adam -se 0 --selection $method --model LinearCLIP --lr 0.01 --weight_decay 0 -sp ./result --batch 128
    python -u main.py --seed $seed  --fraction $fraction --dataset CIFAR100 --data_path ../data --num_exp 1 --workers 4 --optimizer Adam -se 0 --selection $method --model LinearCLIP --lr 0.01 --weight_decay 0 -sp ./result --batch 128
    # python -u main.py --seed $seed  --balance True  --fraction $fraction --dataset CIFAR10 --data_path ../data --num_exp 1 --workers 4 --optimizer SGD -se 0 --selection $method --model SwAVResNet50 --lr 0.1 -sp ./result --batch 128 >> $method.log

    
    done
done
