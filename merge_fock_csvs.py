import os
import pandas as pd

# Path to the root folder
root_dir = "results_fock_experiment"
merged_data = []

# Traverse all subdirectories and collect summary.csv files
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename == "summary.csv":
            file_path = os.path.join(dirpath, filename)
            df = pd.read_csv(file_path)
            merged_data.append(df)

# Concatenate all DataFrames
if merged_data:
    merged_df = pd.concat(merged_data, ignore_index=True)
    merged_df.to_csv("merged_summary.csv", index=False)
    print("Merged file saved as 'merged_summary.csv'")
else:
    print("No summary.csv files found.")
