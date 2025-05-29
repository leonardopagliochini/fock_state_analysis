import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import argparse
import os

# === Parsing argomenti ===
parser = argparse.ArgumentParser(description="Plot heatmap delle best_fidelity da CSV")
parser.add_argument("csv_file", type=str, help="Percorso al file CSV da plottare")
args = parser.parse_args()

# === Carica il CSV ===
if not os.path.exists(args.csv_file):
    raise FileNotFoundError(f"File non trovato: {args.csv_file}")

df = pd.read_csv(args.csv_file)

# === Estrai il numero di Fock ===
df["fock_n"] = df["state_name"].apply(lambda x: int(re.search(r'\d+', x).group()))

# === Crea la pivot table ===
heatmap_data = df.pivot_table(
    index="fock_n",
    columns="n_blocks",
    values="best_fidelity",
    aggfunc="mean"
)

# === Ordina assi ===
heatmap_data = heatmap_data.sort_index(axis=0).sort_index(axis=1)

# === Plot della heatmap ===
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis", linewidths=0.5)
plt.title("Heatmap: Best Fidelity Media per Fock Number e n_blocks")
plt.xlabel("Number of Blocks")
plt.ylabel("Fock Number")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
