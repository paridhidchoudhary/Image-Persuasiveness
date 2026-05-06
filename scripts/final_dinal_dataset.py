import pandas as pd

# Input and output paths
input_path = "datasets/persuasion/persuasion_clip.csv"
output_path = "datasets/persuasion/persuasion_clip_rank.tsv"

# Read the original CSV (comma-separated)
df = pd.read_csv(input_path)

# Rename columns to match RankingAwareCLIP expectations
rename_map = {
    "filepath": "image",
    "title": "caption"
}
df.rename(columns=rename_map, inplace=True)

# Drop unused columns if they exist
drop_cols = ["group", "sample_weights_by_class"]

df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Save as tab-separated file
df.to_csv(output_path, sep="\t", index=False)

print(f"✅ Saved cleaned dataset to: {output_path}")
print(f"Columns: {list(df.columns)}")
