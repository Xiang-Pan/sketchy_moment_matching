import pandas as pd
import numpy as np

# Load the new CSV file
# file_path = 'CIFAR100_TinyNet_df.csv'
file_path = 'CIFAR10_TwoLayerCLIP_df.csv'
try:
    data = pd.read_csv(file_path)
    print("File read successfully!")
except Exception as e:
    print(f"Error reading the file: {e}")
    exit()
# only select the rows with TinyNet
# data = data[data['model'] == 'TinyNet']
data = data[data['model'] == 'TwoLayerCLIP']
data = data[data['dataset'] == 'CIFAR10']
print(data.head())

# Group by method and fraction, then calculate mean and std of accuracy
grouped = data.groupby(['method', 'fraction']).agg(
    mean_acc=('acc', 'mean'),
    std_acc=('acc', 'std'),
    count=('acc', 'size')  # Count the number of samples to calculate std error
).reset_index()

# Convert fraction to integers by multiplying by 50000
grouped['fraction'] = (grouped['fraction'] * 50000).astype(int)

# Calculate standard error
grouped['std_error'] = grouped['std_acc'] / np.sqrt(grouped['count'])

# Pivot the table to get the desired format
pivot_table = grouped.pivot(index='method', columns='fraction', values=['mean_acc', 'std_error'])

# Create a formatted table
formatted_table = pd.DataFrame()
formatted_table['method'] = pivot_table.index

# List of specific fractions to include
specific_fractions = [50, 100, 200, 500, 1000, 2000, 3000, 4000]

# Ensure that only the specific fractions are included in the formatted table
for fraction in specific_fractions:
    mean_col = ('mean_acc', fraction)
    error_col = ('std_error', fraction)
    
    # Construct the formatted string for each cell
    if mean_col in pivot_table.columns and error_col in pivot_table.columns:
        formatted_table[fraction] = [
            f"{row[mean_col] / 100:.5f} ± {row[error_col] / 100:.5f}" if not pd.isna(row[mean_col]) and not pd.isna(row[error_col]) else 'NaN'
            for _, row in pivot_table.iterrows()
        ]
    else:
        formatted_table[fraction] = ['NaN'] * len(formatted_table)

# Reorder columns based on specific fractions
columns_order = ['method'] + specific_fractions
formatted_table = formatted_table[columns_order]

# Generate LaTeX table
latex_table = formatted_table.to_latex(index=False, header=True, column_format='l' + 'l' * len(specific_fractions))

# Display the LaTeX table
print(latex_table)
