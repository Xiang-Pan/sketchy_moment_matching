import wandb

def soft_delete_runs(run):
    run.tags = run.tags + ["deleted"]
    run.update()


def clean_by_name(entity, project):
    """
    Cleans all runs in a project by name, if there are mulple runs with the same name, it will only keep the most recent one and soft-delete the rest (mark them as deleted)
    """
    # it is automatically ranked by the creation time
    runs = wandb.Api().runs(f"{entity}/{project}")
    run_names = set()
    for run in runs:
        if run.name in run_names:
            print(f"soft deleting {run.name} from {run.id}")
            soft_delete_runs(run)
        else:
            run_names.add(run.name)

if __name__ == "__main__":
    clean_by_name(entity="WANDB_ENTITY", project="cov_selection")