import json
import yaml
import matplotlib.pyplot as plt
import os
import seaborn as sns
import click
import hydra
import pandas as pd
from typing import Dict, Any
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def get_value(dict: Dict[str, Any], dot_key: str) -> Any:
    key_path = dot_key.split('.')
    value = dict
    for key in key_path:
        value = value.get(key)
    return value


def load_json_data(json_path):
    """Load data from a JSON file."""
    logger.debug(f"Loading data from {json_path}")
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def extract_values(data):
    for key, value in data.items():
        if isinstance(value, dict) and "value" in value:
            data[key] = value["value"]
    return data

def load_yaml_data(yaml_path):
    """Load data from a YAML file."""
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    data = extract_values(data)
    return data


def plot_data(data_frame, name):
    """Plot data using matplotlib."""
    plt.figure(figsize=(5, 5))
    # Plot each accuracy type
    # # sort by selection fraction
    data_frame.sort_values(by='selection.fraction')
    # pd print
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    data_frame.columns = [col.replace("/", "-") for col in data_frame.columns]

    print(data_frame)
    y_name = "class_unconditioned/use_weights=False/test_as_val=True/tune=False/test/acc"
    y_name = y_name.replace("/","-")
    
    plt.figure(figsize=(5, 5))
    # # sort by selection fraction
    data_frame.sort_values(by='selection.fraction')
    sns.lineplot(data=data_frame, x='selection.fraction', y=y_name, hue='selection.method', marker='o', errorbar="se")
    plt.xlabel('Selection Fraction')
    plt.ylabel('Accuracy')
    plt.title(f"{name} test_as_val=True", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"figures/{name}_test_as_val_True.pdf", bbox_inches='tight')
    print(f"figures/{name}_test_as_val_True.pdf")
    
def get_single_data(config_data, summary_data, key):
    if "+" in key:
        key_list = key.split('+')
        return "+".join([get_single_data(config_data, summary_data, k) for k in key_list])
    else:
        if "acc" in key:
            return summary_data.get(key)
        value = get_value(config_data, key)
        if value is None:
            value = "None"
        return value

def get_name(df):
    dataset_name = df["dataset.name"].unique()[0]
    backbone_name = df["backbone.name+backbone.version"].unique()[0]
    selection_method = "+".join(df["selection.method"].unique())
    return f"{dataset_name}_{backbone_name}_{selection_method}"

def plot_temp():
    # collects/cifar10/clip-vit-base-patch32-ViT-B/32/random/Adam--2/c_conditioned=False
    temp_dir = "collects/cifar10/clip-vit-base-patch32-ViT-B/32/random/Adam--1/c_conditioned=False"
    figure_dir = "figures/cifar10/clip-vit-base-patch32-ViT-B/32/random/Adam--1/c_conditioned=False/"
    keys = ["dataset.name",
            "backbone.name+backbone.version",
            "selection.method",
            "selection.fraction",
            "seed",
            "test/acc_epoch"]
    df_list = []
    for filename in os.listdir(temp_dir):
        if filename.endswith('config.yaml'):
            cfg_hash = filename.split('_')[0]
            results_path = os.path.join(temp_dir, f"{cfg_hash}_results.json")
            config_path = os.path.join(temp_dir, filename)
            config_data = load_yaml_data(config_path)
            json_path = os.path.join(temp_dir, f"{cfg_hash}_wandb-summary.json")
            yaml_path = os.path.join(temp_dir, filename)
            summary_data = load_json_data(json_path)
            config_data = load_yaml_data(yaml_path)
            run_df = {key: get_single_data(config_data, summary_data, key) for key in keys}
            df_list.append(run_df)
    if len(df_list) == 0:
        print("No data found.")
        return
    df = pd.DataFrame(df_list)
    df = df.sort_values(by='selection.fraction')
    # name = get_name(df) + f"_{ft_name}"
    # if "/" in name:
        # name = name.replace("/", "-")
    # df.to_csv(f"figures/{name}.csv", index=False)
    os.makedirs(figure_dir, exist_ok=True)
    with open(f"{figure_dir}/collect.csv", "w") as f:
        df.to_csv(f, index=False)
    # plot_data(df, name)


@hydra.main(config_path="configs", config_name="plot", version_base="1.3.0")
def main(cfg):
    try:
        dataset = [cfg.dataset.name]
        backbone = [cfg.backbone.name + "+" + cfg.backbone.version]
        method = cfg.method
    except:
        dataset = None
        backbone = None
        method = None
    if dataset and backbone and method:
        print(dataset, backbone, method)
        df_list = []
        for m in method:
            for d in dataset:
                for b in backbone:
                    name = f"{d}_{b}_{m}"
                    if "/" in name:
                        name = name.replace("/", "-")
                    df = pd.read_csv(f"figures/{name}.csv")
                    df_list.append(df)
        df = pd.concat(df_list)
        name = get_name(df)
        plot_data(df, name)
    else:
        plot_temp()


def test():
    path = './temp/f4a4e95e596aa57454b6fc90568bee97_config.yaml'
    config_data = load_yaml_data(path)
    d = {"dataset.name": get_single_data(config_data, None, "dataset.name")}
    print(config_data)
    print(d)


if __name__ == "__main__":
    main()
    # test()