import wandb
from tqdm import tqdm
from joblib import Parallel, delayed

# Define a function to delete "diff.patch" files from a single run
def delete_diff_patch(run_id):
    api = wandb.Api()
    run = api.run(f"WANDB_ENTITY/test_time_adaptation/{run_id}")
    files = run.files()
    for file in files:
        if "diff.patch" in file.name:
            file.delete()

# Generator function to yield run IDs
def get_run_ids():
    api = wandb.Api()
    runs = api.runs("WANDB_ENTITY/test_time_adaptation")
    for run in runs:
        yield run.id

runs = wandb.Api().runs("WANDB_ENTITY/test_time_adaptation")

# logger.info(f"Deleting diff.patch files from {len(run_ids)} runs...")
print("Fetching run IDs and deleting diff.patch files...")
# Parallelize the deletion across multiple run IDs
Parallel(n_jobs=-1)(delayed(delete_diff_patch)(run_id) for run_id in tqdm(get_run_ids(), total=len(runs)))
