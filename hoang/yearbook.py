import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np

class YearPredictionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.targets = []

        for folder in ['M', 'F']:
            folder_path = os.path.join(root_dir, folder)
            file_names = sorted(os.listdir(folder_path))  # Ensure consistent ordering
            for file_name in file_names:
                if file_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(folder_path, file_name))
                    year = int(file_name.split('_')[0])
                    self.targets.append(year)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target

def save_split_indices(train_indices, val_indices, file_path):
    np.savez(file_path, train_indices=train_indices, val_indices=val_indices)

def load_split_indices(file_path):
    npzfile = np.load(file_path)
    return npzfile['train_indices'], npzfile['val_indices']

# Define transforms (e.g., resize, normalize)
transform = transforms.Compose([
    transforms.Resize((128, 128)), # Resize images to 128x128, adjust as needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create dataset
dataset = YearPredictionDataset(root_dir='yearbook', transform=transform)

# File path to save/load indices
split_indices_file = 'yearbook/split_indices.npz'

# Check if the split indices file exists
if os.path.exists(split_indices_file):
    # Load the split indices
    train_indices, val_indices = load_split_indices(split_indices_file)
else:
    # Define train/val split ratio
    train_ratio = 0.8
    val_ratio = 0.2

    # Calculate lengths for training and validation sets
    total_size = len(dataset)
    indices = list(range(total_size))
    np.random.seed(42)  # Set seed for reproducibility
    np.random.shuffle(indices)
    train_size = int(train_ratio * total_size)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    # Save the split indices for future use
    save_split_indices(train_indices, val_indices, split_indices_file)

# Create subset datasets
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
print(f'Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}')
# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Example usage
for images, targets in train_loader:
    print(f'Train batch - images shape: {images.shape}, targets shape: {targets.shape}')
    break

for images, targets in val_loader:
    print(f'Val batch - images shape: {images.shape}, targets shape: {targets.shape}')
    break
