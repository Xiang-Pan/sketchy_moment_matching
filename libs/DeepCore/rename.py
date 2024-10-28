import os

# Define the mappings
mappings = {
    "Uncertainty_Entropy": "Uncertainty-Entropy",
    "Uncertainty_Margin": "Uncertainty-Margin",
    "Uncertainty_LeastConfidence": "Uncertainty-LeastConfidence"
}

# Get the current directory
# current_directory = os.getcwd()
# LABROOTs/data_pruning/libs/DeepCore/result/test_as_val_True_balance_false
dir = "result/test_as_val_True_balance_false"
# Iterate over each file in the directory
for filename in os.listdir(dir):
    # Check if the filename contains any of the keys in mappings
    for old_name, new_name in mappings.items():
        if old_name in filename:
            # Generate the new filename
            new_filename = filename.replace(old_name, new_name)
            # Rename the file
            print(f"Renaming {filename} to {new_filename}")
            os.rename(os.path.join(dir, filename), os.path.join(dir, new_filename))
            # break