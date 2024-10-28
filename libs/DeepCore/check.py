import hydra
import os
import glob
import re
import torch
import pandas as pd
from rich import print

def clean():
    # get all files under result
    all_files = glob.glob("result/test_as_val_True_balance_False/*")
    # check _ in the file name
    for file in all_files:
        # print(file.count("_"))
        if file.count("_") != 8:
            print(file)
            os.remove(file)

def check_fun(c_conditioned=False):
    if c_conditioned:
        balance = "True"
    else:
        balance = "False"
    output = f"result/test_as_val_True_balance_{balance}"
    ckpts = glob.glob(f"{output}/*.ckpt")
    sample = torch.load(ckpts[0], map_location="cpu")
    print(sample["exp"])
    print(sample["subset"].keys())
    print(sample["sel_args"])
    # tasks = ["uniform", "cd", "glister", "grand", "herding", "forgetting", "deepfool", "entropy", "margin", "leastconfidence"]
    # fractions = [0.001, 0.01, 0.02, 0.04, 0.06, 0.08]
    # models = ["ResNet18", "ResNet50"]
    # method_rename = {
    #     "uniform": "Uniform",
    #     "cd": "ContextualDiversity",
    #     "glister": "Glister",
    #     "grand": "GraNd",
    #     "herding": "Herding",
    #     "forgetting": "Forgetting",
    #     "deepfool": "DeepFool",
    #     "entropy": "Entropy",
    #     "margin": "Margin",
    #     "leastconfidence": "LeastConfidence"
    # }
    # result/CIFAR10_ResNet18_ContextualDiversity_exp0_epoch0_2024-04-23 10:26:52.870288_0.1_0.000000.ckpt
    filenames = [os.path.basename(ckpt) for ckpt in ckpts]
    print(filenames[0])
    dataset, model, method, exp, epoch, timestamp, fraction, seed = filenames[0].split("_")
    dataset_list = ["StanfordCars", "cifar10"]
    method_list = ["Uniform", "ContextualDiversity", "Glister", "GraNd", "Herding", "Forgetting", "DeepFool", "Uncertainty-Entropy", "Uncertainty-Margin", "Uncertainty-LeastConfidence"]
    model_list = ["LinearCLIP", "TwoLayerCLIP", "ThreeLayerCLIP", "LinearResNet18", "TwoLayerResNet18"]
    dataset_name = "-".join(dataset_list)
    model_name = "-".join(model_list)
    fraction_list = [
        0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08] + [0.0614, 0.1228, 0.1842, 0.2456, 0.3070, 0.3684, 0.4298, 0.4912]
    fraction_list = [str(f) for f in fraction_list]
    seed_list = [0, 1, 2, 3, 4]
    seed_list = [str(s) for s in seed_list]
    t_list = []
    t2path = {}
    df = pd.DataFrame(columns=["dataset", "model", "method", "fraction", "seed", "acc", "timestamp", "file"])
    for file in filenames:
        # print(file)
        if len(file.split("_")) != 8:
            print(file)
            continue
        dataset, model, method, exp, epoch, timestamp, fraction, seed = file.split("_")
        # acc = acc.replace(".ckpt", "")
        seed = seed.replace(".ckpt", "")
        seed = seed[0]
        
        acc = 0
        if acc == "unknown":
            continue
        acc = float(acc)
        t = (dataset, model, method, fraction, seed)
        path = f"result/test_as_val_True_balance_False/{file}"
        df = pd.concat([df, pd.DataFrame([[dataset, model, method, fraction, seed, acc, timestamp, path]], columns=df.columns)], ignore_index=True)
        t2path[t] = f"libs/DeepCore/result/test_as_val_True_balance_{balance}/{file}"
    # df.to_csv("df.csv", index=False)
    # for (dataset, model, method, fraction, seed) keep the latest timestamp
    df = df.sort_values(by=["dataset", "model", "method", "fraction", "seed", "timestamp"])
    df = df.drop_duplicates(subset=["dataset", "model", "method", "fraction", "seed"], keep="last")
    print(dataset_name)
    df.to_csv(f"{dataset_name}_{model_name}_df.csv", index=False)
    print(f"{dataset_name}_{model_name}_df.csv")
    
    # save t2path
    import pickle as pkl
    with open(f"{balance}_t2path.pkl", "wb") as f:
        pkl.dump(t2path, f)
    # write txt
    with open(f"{balance}_t2path.txt", "w") as f:
        for t, path in t2path.items():
            f.write(f"{t} {path}\n")
    print(f"{balance}_t2path.txt")
    from itertools import product
    miss_list = []
    for dataset, model, method, fraction, seed in product(dataset_list, model_list, method_list, fraction_list, seed_list):
        t = (dataset, model, method, fraction, seed)
        if t not in t2path:
            miss_list.append(t)
    with open("miss_list.txt", "w") as f:
        for t in miss_list:
            f.write(f"{t}\n")
        

if __name__ == "__main__":
    print("check False")
    check_fun(c_conditioned=False)
    print("check True")
    # check_fun(c_conditioned=True)