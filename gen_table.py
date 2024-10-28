import pandas as pd
import numpy as np
import wandb
from rich import print
import joblib
from joblib import Memory

cache_dir = "cache"
memory = Memory(cache_dir, verbose=1)



def prepare_formatted_output(df, fractions=[50, 100, 200, 500, 1000, 2000, 3000, 4000]):
    method = df["selection_method"].iloc[0]
    
    # Drop fully NaN columns
    df = df.dropna(axis=1, how="all")
    
    # Drop unnecessary columns
    df = df.drop(columns=["dataset", "selection_method", "backbone", "seed"])
    if "selection.c_conditioned" in df.columns:
        df = df.drop(columns=["selection.c_conditioned"])
    # Drop rows with NaN values
    df = df.dropna()

    # Calculate the mean and standard error for each group
    grouped = df.groupby("fraction")
    means = grouped.mean()
    std_errs = grouped.std() / np.sqrt(grouped.count())

    # Combine mean and std_err into the desired format
    combined = means.applymap(lambda x: f"{x:.4f}") + " ± " + std_errs.applymap(lambda x: f"{x:.4f}")

    # Ensure all specified fractions are included
    for fraction in fractions:
        if fraction not in combined.index:
            combined.loc[fraction] = np.nan

    # Sort the index to maintain the order of fractions
    combined = combined.loc[fractions]

    # Pivot the DataFrame to make fraction as columns
    pivoted = combined.T

    # # Prepare the formatted output
    # formatted_output = []
    # for index, row in pivoted.iterrows():
    #     formatted_row = " & ".join(row.fillna("NaN"))
    #     formatted_output.append(f"{index} & {formatted_row} \\\\")
    
    formatted_output = [f"\\midrule\nmethod & " + " & ".join(map(str, fractions)) + " \\\\",
                        "\\midrule"]
    for index, row in pivoted.iterrows():
        formatted_row = " & ".join(row.fillna("NaN"))
        formatted_output.append(f"{method} & {index} & {formatted_row} \\\\")
    # Add header row with fractions
    # formatted_output.insert(0, "\\midrule")
    # header_row = f"{method} & " + " & " * (len(fractions) - 1) + " \\\\"
    # formatted_output.insert(0, header_row)
    # formatted_output.insert(0, "\\midrule")

    # Print the formatted output
    final_output = "\n".join(formatted_output)
    # Replace "_" with "\_" to avoid LaTeX error
    final_output = final_output.replace("_", "\\_")
    return final_output


@memory.cache
def fetch_run_data(dataset_name, selection_method, backbone_name, fractions, seeds, addition_filters={}, stage="selection"):
    
    """
    Fetches run data from Weights and Biases (wandb) API based on the provided filters.

    Args:
        dataset_name (str): The name of the dataset.
        selection_method (str): The selection method used.
        backbone_name (str): The backbone model name.
        fractions (list): A list of selection fractions to query.
        seeds (list): A list of seeds to query.

    Returns:
        list: A list of dictionaries containing the run data.
    """
    df = []
    for fraction in fractions:
        for seed in seeds:
            filters = {
                "config.dataset.name": {"$in": [dataset_name]},
                "config.selection.method": {"$in": [selection_method]},
                "config.backbone.name": {"$in": [backbone_name]},
                "config.seed": {"$in": [seed]},
                "config.selection.fraction": {"$in": [fraction]},
            }
            filters.update(addition_filters)
            runs = wandb.Api().runs(f"WANDB_ENTITY/data_pruning-{stage}", filters)
            if runs:
                run = runs[0]
                if stage == "selection":
                    c_true_acc = run.summary.get("class_conditioned/use_weights=False/test_as_val=True/tune=False/test/acc", np.nan)
                    sample_acc = run.summary.get("sampled_0/use_weights=False/test_as_val=True/tune=False/test/acc", np.nan)
                    c_false_acc = run.summary.get("class_unconditioned/use_weights=False/test_as_val=True/tune=False/test/acc", np.nan)
                    c_true_f1 = run.summary.get("class_conditioned/use_weights=False/test_as_val=True/tune=False/test/f1", np.nan)
                    sample_f1 = run.summary.get("sampled_0/use_weights=False/test_as_val=True/tune=False/test/f1", np.nan)
                    c_false_f1 = run.summary.get("class_unconditioned/use_weights=False/test_as_val=True/tune=False/test/f1", np.nan)
                    df.append({
                        "dataset": dataset_name,
                        "selection_method": selection_method,
                        "backbone": backbone_name,
                        "fraction": fraction,
                        "seed": seed,
                        "c_true_acc": c_true_acc,
                        "sample_acc": sample_acc,
                        "c_false_acc": c_false_acc,
                        "c_true_f1": c_true_f1,
                        "sample_f1": sample_f1,
                        "c_false_f1": c_false_f1
                    })
                elif stage == "finetuning":
                    # c_conditioned = run.config.get("selection")["c_conditioned"]
                    test_acc = run.summary.get("test/acc", np.nan)
                    test_f1 = run.summary.get("test/f1", np.nan)
                    df.append({
                        "dataset": dataset_name,
                        "selection_method": selection_method,
                        "backbone": backbone_name,
                        "fraction": fraction,
                        "seed": seed,
                        # "selection.c_conditioned": c_conditioned,
                        "test_acc": test_acc,
                        "test_f1": test_f1
                    })
    df = pd.DataFrame(df)
    return df


def main():
    df = fetch_run_data("StanfordCars", "random", "clip-vit-base-patch32", [50, 100, 200, 500, 1000, 2000, 3000, 4000], [0, 1, 2, 3, 4])
    print(prepare_formatted_output(df, fractions=[500, 1000, 2000, 3000, 4000]))

if __name__ == "__main__":
    main()