import pandas as pd

# Load the CSV file
df_path = 'figures/cifar10/clip-vit-base-patch32-ViT-B/32/random/Adam--1/c_conditioned=False/collect.csv'
df = pd.read_csv(df_path)

key_name = "test/acc_epoch"

# Group by 'selection.method' and 'selection.fraction' and calculate the mean and standard error
df_group = df.groupby(['selection.method', 'selection.fraction']).agg(
    mean_acc=(key_name, 'mean'),
    std_error=(key_name, lambda x: x.std() / (len(x) ** 0.5))
).reset_index()

# Combine mean and standard error into a single string
df_group['mean_std'] = df_group.apply(lambda row: f'{row["mean_acc"]:.5f} ± {row["std_error"]:.5f}', axis=1)

# Pivot the table to get the desired format
pivot_table = df_group.pivot(index='selection.method', columns='selection.fraction', values='mean_std')

# Sort the columns to ensure correct order
pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)

# Prepare the LaTeX table
latex_table = pivot_table.to_latex()

# Print LaTeX table
print(latex_table)
