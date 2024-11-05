#!/bin/bash
#SBATCH --job-name=ml-selection-experiments
#SBATCH --output=./logs/slurm/result_%A_%a.out
#SBATCH --error=./logs/slurm/result_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --time=240
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --array=0-799

# Define tasks, fractions, models, and seeds
#* 10 tasks
TASKS=("uniform" "cd" "glister" "grand" "herding" "forgetting" "deepfool" "entropy" "margin" "leastconfidence")
# FRACTIONS=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08)
# 1000 / 50000 = 0.02
#* 8 fractions
FRACTIONS=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08)
#* 2 model
MODELS=("LinearCLIP")
#* 5 seeds
SEEDS=(0 1 2 3 4)

dataset="cifar10"
# total number of jobs = 10 * 8 * 2 * 5 = 800

# Total number of combinations per task
COMBINATIONS_PER_TASK=$((${#FRACTIONS[@]} * ${#MODELS[@]} * ${#SEEDS[@]}))

# Determine task, fraction, model, and seed from SLURM_ARRAY_TASK_ID
TASK_INDEX=$((SLURM_ARRAY_TASK_ID / $COMBINATIONS_PER_TASK))
INNER_INDEX=$((SLURM_ARRAY_TASK_ID % $COMBINATIONS_PER_TASK))
FRACTION_INDEX=$((INNER_INDEX / (${#MODELS[@]} * ${#SEEDS[@]})))
MODEL_SEED_INDEX=$((INNER_INDEX % (${#MODELS[@]} * ${#SEEDS[@]})))
MODEL_INDEX=$((MODEL_SEED_INDEX / ${#SEEDS[@]}))
SEED_INDEX=$((MODEL_SEED_INDEX % ${#SEEDS[@]}))

SELECTED_TASK=${TASKS[$TASK_INDEX]}
SELECTED_FRACTION=${FRACTIONS[$FRACTION_INDEX]}
SELECTED_MODEL=${MODELS[$MODEL_INDEX]}
SELECTED_SEED=${SEEDS[$SEED_INDEX]}
balance=False
# Load any necessary modules or source your .bashrc here
# module load python/3.8
# source ~/env/bin/activate

# Echo starting time
echo "Job started on $(date)"
echo "Running task: $SELECTED_TASK with model: $SELECTED_MODEL, fraction: $SELECTED_FRACTION, seed: $SELECTED_SEED"

# Run the task
task $SELECTED_TASK model=$SELECTED_MODEL fraction=$SELECTED_FRACTION seed=$SELECTED_SEED test_as_val=True dataset=$dataset balance=$balance

# Echo ending time
echo "Job ended on $(date)"
