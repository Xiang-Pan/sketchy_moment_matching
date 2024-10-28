import os

def rsync(source, target):
    print("source:", source)
    print("target:", target)
    os.system(f"rsync -avz --progress {source} {target}")

if __name__ == "__main__":
    relative_path = "cached_datasets/backbone#name=resnet50-version=IMAGENET1K_V2-classifier=linear"
    source_server = "ds"
    target_server = "ai"
    source_path = f"LABROOTs/data_pruning/{relative_path}"
    target_path = f"LABROOTs/data_pruning/{relative_path}"
    # check if
    server = os.environ["server"]
    source = f"{source_server}:{source_path}"
    target = f"{target_server}:{target_path}"
    if source_server == server:
        source = source_path
    if target_server == server:
        target = target_path
    rsync(source, target)