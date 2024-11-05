#!/bin/bash
#SBATCH --job-name=ml-selection-experiments
#SBATCH --output=./logs/slurm/result_%A_%a.out
#SBATCH --error=./logs/slurm/result_%A_%a.err
#SBATCH --gres=gpu:a100:1
#SBATCH --time=240
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=general
#SBATCH --array=0-149

# Define tasks, fractions, models, and seeds
TASKS=("glister" "deepfool")
# 3 tasks
FRACTIONS=(0.0614 0.1228 0.2456 0.3684 0.4912)
# 5 fractions
MODELS=("LinearCLIP" "TwoLayerCLIP" "ThreeLayerCLIP")
# 2 model
SEEDS=(0 1 2 3 4)
# 5 seeds
# total number of jobs = 3 * 5 * 2 * 5 = 150

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
