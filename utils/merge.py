import os
import shutil

def recursive_copy(source, destination):
    if not os.path.exists(destination):
        os.makedirs(destination, exist_ok=True)

    for item in os.listdir(source):
        source_item = os.path.join(source, item)
        destination_item = os.path.join(destination, item)

        # Skip copying last.ckpt files
        if os.path.basename(source_item) == 'last.ckpt':
            continue

        if os.path.islink(source_item):
            if os.path.exists(destination_item):
                os.remove(destination_item)
            link_target = os.readlink(source_item)
            os.symlink(link_target, destination_item)
        elif os.path.isdir(source_item):
            recursive_copy(source_item, destination_item)
        else:
            if os.path.exists(destination_item):
                os.remove(destination_item)
            shutil.copy2(source_item, destination_item)

source_dir = "cached_datasets/backbone#name=resnet50-version=IMAGENET1K_V2"
target_dir = "cached_datasets/backbone#name=resnet50-version=IMAGENET1K_V2-classifier=linear"
# Copy everything from source_dir to dest_dir recursively, excluding last.ckpt files
recursive_copy(source_dir, target_dir)

# Move the source directory to a new location named "back"
back_dir = 'back'  # Adjust this path as needed
os.makedirs(back_dir, exist_ok=True)
shutil.move(source_dir, os.path.join(back_dir, os.path.basename(source_dir)))
