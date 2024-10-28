import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from joblib import Memory
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score

cache_location = "./cache"
memory = Memory(cache_location, verbose=1)

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def test_coverage(X, X_sel, name, epoch=-1):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    if isinstance(X_sel, np.ndarray):
        X_sel = torch.from_numpy(X_sel)
    X = X.cpu()
    X_sel = X_sel.cpu()
    Cov_X = X.T @ X / X.shape[0]
    Cov_X_sel = X_sel.T @ X_sel / X_sel.shape[0]
    relative_cov_gap_fro = torch.linalg.norm(Cov_X - Cov_X_sel, ord='fro') / torch.linalg.norm(Cov_X, ord='fro')
    wandb.log({f"{name}/relative_cov_gap_fro": relative_cov_gap_fro})
    logger.debug(f"relative_cov_gap_fro: {relative_cov_gap_fro}")
    e, v = torch.linalg.eig(Cov_X)
    e = e.real
    v = v.real
    sel_e = torch.diag(v.T @ Cov_X_sel @ v)
    logger.debug(f"sel_e.shape: {sel_e.shape}")
    # plot the e and sel_e
    plt.figure()
    x_list = list(range(len(e)))
    plt.plot(x_list, e, label="e")
    plt.plot(x_list, sel_e, label="sel_e")
    # y log
    plt.yscale("log")
    plt.legend()
    wandb_dir = os.path.dirname(wandb.run.dir)
    if wandb_dir == "/":
        wandb_dir = "."
    image_path = f"{wandb_dir}/{name}_eign_dist.pdf"
    image_dir = os.path.dirname(image_path)
    logger.debug(f"image_dir: {image_dir}")
    os.makedirs(image_dir, exist_ok=True)
    plt.savefig(f"{wandb_dir}/{name}_eign_dist.pdf")
    wandb.log({f"{name}/eign_dist": wandb.Image(plt), "epoch": epoch})
    logger.critical(image_path)
    plt.close()
    return relative_cov_gap_fro


def test_selection(cfg, idxes, eb_dataset_dict, unpretrained_eb_dataset_dict, name, task, test_as_val, seed=None):
    if seed is None:
        seed = cfg.seed
    is_pretrained = hasattr(cfg, "pretraining")
    logger.info("is_pretrained: %s", is_pretrained)
    dataset_name = cfg.dataset.name
    if is_pretrained:
        test_eb(eb_dataset_dict, idxes, weights=None, use_weights=False, name=f"pretrained/{name}", task=task, seed=seed, dataset_name=dataset_name, test_as_val=test_as_val)
        test_eb(unpretrained_eb_dataset_dict, idxes, weights=None, use_weights=False, name=f"{name}", task=task, seed=seed, dataset_name=dataset_name, test_as_val=test_as_val)
    else:
        test_eb(eb_dataset_dict, idxes, weights=None, use_weights=False, name=f"{name}", task=task, seed=seed, dataset_name=dataset_name, test_as_val=test_as_val)

@memory.cache
def test_eb(dataset_dict,
            idxes,
            weights,
            seed,
            dataset_name,
            use_weights=False,
            name="random",
            use_mlp=False,
            task="regression",
            test_as_val=True,
            epoch=-1,
            tune=True,
            ):
    """_summary_: test the eb of the idxes
    Args:
        dataset_dict (_type_): _description_
        idxes (_type_): _description_
        weights (_type_): _description_
        use_weights (bool, optional): _description_. Defaults to False.
        name (str, optional): _description_. Defaults to "random".

    Returns:
        _type_: _description_
    """
    # if tune == False:
        # assert test_as_val == False, "tune=False, test_as_val=True is not allowed"
    if type(idxes) == torch.Tensor:
        idxes = idxes.cpu().numpy()

    assert seed is not None
    logger.info(f"len(idxes): {len(idxes)}")
    dataset_device = dataset_dict["train"]["X"].device
    # make sure everything is on the same device
    if task == "classification":
        model = MLPClassifier(hidden_layer_sizes=(), activation="identity") if use_mlp else LogisticRegression(random_state=seed)
    else:
        model = MLPRegressor(hidden_layer_sizes=(), activation="identity", random_state=seed) if use_mlp else Ridge(random_state=seed)
    model = Pipeline([("scaler", StandardScaler()), ("clf", model)], memory="cache", verbose=True)
    if use_mlp:
        param_grid = {
            "clf__alpha": [1e-4, 1e-3, 1e-2, 5e-2, 1e-1],
        }
    else:
        if task == "classification":
            #! Default value
            if tune:
                param_grid = {
                    "clf__C": [1e-4, 1e-3, 1e-2, 1e-1, 1e0],
                    "clf__solver": ["lbfgs"],
                    "clf__max_iter": [100],
                    "clf__class_weight": [None],
                }
            else:
                param_grid = {
                    "clf__C": [1e-2],
                    "clf__solver": ["lbfgs"],
                    "clf__max_iter": [100],
                    "clf__class_weight": [None],
                }
        else:
            param_grid = {
                "clf__alpha": [1e-4, 1e-3, 1e-2, 5e-2, 1e-1],
            }
    # log the param_grid
    wandb.log({f"{name}/param_grid": param_grid, "epoch": epoch})
    train_full_data = dataset_dict["train"]["X"].cpu().numpy()
    train_data = train_full_data[idxes]
    train_full_targets = dataset_dict["train"]["Y"].cpu().numpy()
    train_targets = train_full_targets[idxes]
    logger.info(f"sel train_data: {train_data.shape}")
    logger.info(f"sel train_targets: {train_targets.shape}")
    # make sure train_targets have more than one class
    if task == "classification":
        if len(np.unique(train_targets)) == 1:
            logger.critical("train_targets distribution: %s", np.unique(train_targets, return_counts=True))
            return
        logger.info("train_targets distribution: %s", np.unique(train_targets, return_counts=True))
    test_data = dataset_dict["test"]["X"].cpu().numpy()
    test_targets = dataset_dict["test"]["Y"].cpu().numpy()
    if test_as_val:
        val_data = test_data
        val_targets = test_targets
        train_val_data = np.concatenate([train_data, test_data], axis=0)
        train_val_targets = np.concatenate([train_targets, test_targets], axis=0)
    else:
        val_data = dataset_dict["val"]["X"].cpu().numpy()
        val_targets = dataset_dict["val"]["Y"].cpu().numpy()
        train_val_data = np.concatenate([train_data, val_data], axis=0)
        train_val_targets = np.concatenate([train_targets, val_targets], axis=0)
    #! MODIFIED here!
    val_size = len(val_data)
    test_fold = [-1] * len(idxes) + [0] * val_size
    logger.info(f"len(idxes): {len(idxes)}")
    logger.info(f"val_size: {val_size}")
    if use_weights:
        train_val_weights = np.ones_like(train_val_targets)
        train_val_weights[:len(idxes)] = weights

    ps = PredefinedSplit(test_fold=test_fold)
    grid_search = GridSearchCV(model, param_grid, cv=ps, n_jobs=-1, verbose=1, refit=False)
    if use_weights:
        grid_search.fit(train_val_data, train_val_targets, clf__sample_weight=train_val_weights)
    else:
        grid_search.fit(train_val_data, train_val_targets)
    logger.info(grid_search.best_params_)
    logger.info(grid_search.best_score_)
    # refit the best model
    best_model = model.set_params(**grid_search.best_params_)
    best_model.fit(train_data, train_targets)


    if task == "classification":
        from sklearn.metrics import f1_score
        from functools import partial
        f1_score_partial = partial(f1_score, average="macro")
        metric_dict = {
            "acc": accuracy_score,
            "f1": f1_score_partial
        }
    elif task == "regression":
        metric_dict = {
            "mse": mean_squared_error
        }

    test_pred = best_model.predict(test_data)
    train_pred = best_model.predict(train_val_data)
    val_pred = test_pred if test_as_val else best_model.predict(val_data)

    for metric_name, metric in metric_dict.items():
        test_metric = metric(test_targets, test_pred)
        train_metric = metric(train_val_targets, train_pred)
        val_metric = metric(val_targets, val_pred)
        wandb.log({f"{name}/use_weights={use_weights}/test_as_val={test_as_val}/tune={tune}/train/{metric_name}": train_metric,
                   f"{name}/use_weights={use_weights}/test_as_val={test_as_val}/tune={tune}/val/{metric_name}": val_metric,
                   f"{name}/use_weights={use_weights}/test_as_val={test_as_val}/tune={tune}/test/{metric_name}": test_metric,
                   f"epoch": epoch
                   })
        logger.info(f"{name}/use_weights={use_weights}/test_as_val={test_as_val}/tune={tune}/test/{metric_name}: {test_metric}")

    logger.debug("train_full_data: %s", train_full_data.shape)
    logger.debug("idxes: %s", idxes.shape)
    logger.debug("max(idxes): %s", max(idxes))
    test_coverage(train_full_data, train_data, name=f"{name}/train", epoch=epoch)
    wandb_dir = wandb.run.dir
    wandb_dir = os.path.dirname(wandb_dir)
    from utils.viz import plot_tsne
    # We do not plot_tsne for faster sweep
    # plot_tsne(X=train_full_data, Y=train_full_targets, selected_idxes=idxes, dataset_name=dataset_name, wandb_dir=wandb_dir, seed=seed, epoch=epoch)

    for k, v in grid_search.best_params_.items():
        if "__" in k:
            # For logging the best hyperparameters
            wandb.log({f"{name}/use_weights={use_weights}/test_as_val={test_as_val}/best_params/{k}": v, "epoch": epoch})
    return best_model

def linear_probe(c_unconditioned_idxes, c_unconditioned_weights, c_conditioned_idxes, c_conditioned_weights, dataset_dict, seed, dataset_name, debug=True):
    if debug:
        test_eb(
            dataset_dict=dataset_dict,
            idxes=c_unconditioned_idxes,
            weights=c_unconditioned_weights,
            seed=seed,
            use_weights=False,
            name="class_unconditioned",
            use_mlp=False,
            task="classification",
            dataset_name=dataset_name,
            test_as_val=False,
            tune=False
        )
        test_eb(
            dataset_dict=dataset_dict,
            idxes=c_conditioned_idxes,
            weights=c_conditioned_weights,
            seed=seed,
            use_weights=False,
            name="class_conditioned",
            use_mlp=False,
            task="classification",
            dataset_name=dataset_name,
            test_as_val=False,
            tune=False
        )
        return
    # check
    for tune in [True, False]:
        for test_as_val in [True]:
            if c_unconditioned_idxes is not None:
                test_eb(
                    dataset_dict=dataset_dict,
                    idxes=c_unconditioned_idxes,
                    weights=None,
                    seed=seed,
                    use_weights=False,
                    name="class_unconditioned",
                    use_mlp=False,
                    task="classification",
                    dataset_name=dataset_name,
                    test_as_val=test_as_val,
                    tune=tune
                )
                if c_unconditioned_weights is not None:
                    test_eb(
                        dataset_dict=dataset_dict,
                        idxes=c_unconditioned_idxes,
                        weights=c_unconditioned_weights,
                        seed=seed,
                        use_weights=True,
                        name="class_unconditioned",
                        use_mlp=False,
                        task="classification",
                        dataset_name=dataset_name,
                        test_as_val=test_as_val,
                        tune=tune
                    )
            if c_conditioned_idxes is not None:
                test_eb(
                    dataset_dict=dataset_dict,
                    idxes=c_conditioned_idxes,
                    weights=None,
                    seed=seed,
                    use_weights=False,
                    name="class_conditioned",
                    use_mlp=False,
                    task="classification",
                    dataset_name=dataset_name,
                    test_as_val=test_as_val,
                    tune=tune
                )
                if c_conditioned_weights is not None:
                    test_eb(
                        dataset_dict=dataset_dict,
                        idxes=c_conditioned_idxes,
                        weights=c_conditioned_weights,
                        seed=seed,
                        use_weights=True,
                        name="class_conditioned",
                        use_mlp=False,
                        task="classification",
                        dataset_name=dataset_name,
                        test_as_val=test_as_val,
                        tune=tune
                    )

import json
def check_finished(hash_dir):
    if not os.path.exists(f"{hash_dir}/c_unconditioned_idxes.pth"):
        logger.info(f"{hash_dir}/c_unconditioned_idxes.pth not found")
        return False
    key_list = [
        "class_unconditioned/use_weights=False/test_as_val=True/tune=True/test/acc"
    ]
    summary_file = f"{hash_dir}/wandb/latest-run/files/wandb-summary.json"
    # minio_client = get_minio_client()

    if os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            summary = json.load(f)
        if all([k in summary for k in key_list]):
            logger.info(summary)
            return True
        logger.info(f"{summary_file} does not contain all required keys.")
    # elif exists_on_minio(minio_client, bucket_name="labs", object_name=summary_file):
    #     # Download the summary file to a local temporary file for processing
    #     local_temp_file = f"temp_summary.json"
    #     download_file(minio_client, summary_file, local_temp_file)

    #     # Read and process the JSON file
    #     with open(local_temp_file, "r") as f:
    #         summary = json.load(f)
    #     os.remove(local_temp_file)  # Clean up the local temporary file

    #     # Check if all required keys are in the summary
    #     if all([k in summary for k in key_list]):
    #         return True
    #     else:
    #         missing_keys = set(key_list) - set(summary.keys())
    #         print(f"{summary_file} does not contain all required keys.")
    #         print(f"Missing keys: {missing_keys}")
    #         return False
    else:
        logger.info(f"{summary_file} does not exist in remote storage.")
        return False

@hydra.main(config_path="configs", config_name="default", version_base="1.3.0")
def main(cfg):
    from omegaconf import OmegaConf
    from utils.data_utils import load_eb_dataset_cfg
    from utils.hash_utils import get_cfg_hash
    dataset_dict = load_eb_dataset_cfg(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_hash = get_cfg_hash(cfg_dict)
    global wandb_dir
    wandb_dir = f"outputs/selection/{cfg_hash}"
    output_dir = wandb_dir

    if check_finished(output_dir):
        logger.info(f"{output_dir} is finished")
        return
    wandb.init(project="data_pruning-linear_probing",
            entity="WANDB_ENTITY",
            name=cfg_hash,
            config=cfg_dict, # type: ignore
            mode="online" if not cfg.debug else "disabled",
            tags=["0.06"],
            dir=f"outputs/selection/{cfg_hash}"
            )
    c_conditioned_idxes = torch.load(f"{output_dir}/c_conditioned_idxes.pth")
    # c_conditioned_weights = torch.load(f"{output_dir}/c_conditioned_weights.pth")
    c_unconditioned_idxes = torch.load(f"{output_dir}/c_unconditioned_idxes.pth")
    # c_unconditioned_weights = torch.load(f"{output_dir}/c_unconditioned_weights.pth")
    seed = cfg.seed
    for tune in [True, False]:
        test_eb(
            dataset_dict=dataset_dict,
            idxes=c_unconditioned_idxes,
            weights=None,
            seed=seed,
            use_weights=False,
            name="class_unconditioned",
            use_mlp=False,
            task="classification",
            dataset_name=dataset_name,
            test_as_val=True,
            tune=tune
        )
        test_eb(
            dataset_dict=dataset_dict,
            idxes=c_conditioned_idxes,
            weights=None,
            seed=seed,
            use_weights=False,
            name="class_conditioned",
            use_mlp=False,
            task="classification",
            dataset_name=dataset_name,
            test_as_val=True,
            tune=tune
        )


if __name__ == "__main__":
    main()