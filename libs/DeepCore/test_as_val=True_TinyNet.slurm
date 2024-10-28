#!/bin/bash
#SBATCH --job-name=ml-selection-experiments
#SBATCH --output=./logs/slurm/result_%A_%a.out
#SBATCH --error=./logs/slurm/result_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --time=240
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=cdac-contrib
#SBATCH --array=0-359

# Define tasks, fractions, models, and seeds
# TASKS=("uniform" "cd" "glister" "grand" "herding" "forgetting" "deepfool" "entropy" "margin" "leastconfidence")
# TASKS=("uniform")
TASKS=("uniform" "cd" "glister" "grand" "herding" "forgetting" "deepfool" "entropy" "margin" "leastconfidence")
# 9 tasks
FRACTIONS=(0.001 0.002 0.004 0.01 0.02 0.04 0.06 0.08)
# 8 fractions
MODELS=("TinyNet")
# 1 model
SEEDS=(0 1 2 3 4)
# 5 seeds
# total number of jobs = 9 * 8 * 1 * 5 = 360

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

# Load any necessary modules or source your .bashrc here
# module load python/3.8
# source ~/env/bin/activate

# Echo starting time
echo "Job started on $(date)"
echo "Running task: $SELECTED_TASK with model: $SELECTED_MODEL, fraction: $SELECTED_FRACTION, seed: $SELECTED_SEED"

# Run the task
task $SELECTED_TASK model=$SELECTED_MODEL fraction=$SELECTED_FRACTION seed=$SELECTED_SEED test_as_val=True

# Echo ending time
echo "Job ended on $(date)"
