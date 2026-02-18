# This script processes the AMR summary dataset to create a feature matrix for machine learning. 
# It extracts gene presence/absence features, computes resistance breadth and MDR labels, and saves the final feature table for modeling.
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "processed" / "amr_summary_dataset.csv"
OUT  = BASE / "data" / "processed_amr_features.csv"

df = pd.read_csv(DATA)

#  Gene features are extracted from the "AMR_Gene_Profile" column, which contains semicolon-separated gene names. 
genes = df["AMR_Gene_Profile"].fillna("").str.split(";")
# The presence of each gene is encoded as a binary feature using one-hot encoding, resulting in a matrix where each column corresponds to a specific gene and each row indicates whether that gene is present in the isolate.
X_gene = genes.str.join("|").str.get_dummies(sep="|")
# To ensure consistent and clean column names, the gene feature columns are prefixed with "gene_" and any spaces are replaced with underscores.
X_gene.columns = ["gene_" + c.strip().replace(" ", "_") for c in X_gene.columns]

#  Phenotype classes (only for target creation; NOT used as features) are extracted from the "Drug_Resistance_Phenotype" column, which contains semicolon-separated resistance classes.
classes = df["Drug_Resistance_Phenotype"].fillna("").str.split(";")
# Similar to the gene features, the presence of each resistance class is encoded as a binary feature using one-hot encoding, resulting in a matrix where each column corresponds to a specific resistance class and each row indicates whether that class is present in the isolate.
X_class = classes.str.join("|").str.get_dummies(sep="|")
X_class.columns = ["class_" + c.strip().replace(" ", "_") for c in X_class.columns]

# Create resistance breadth and binary MDR label
# The total number of resistance classes for each isolate is calculated by summing the binary class features, resulting in a "total_resistance_classes" feature that represents the breadth of resistance.
df["total_resistance_classes"] = X_class.sum(axis=1)
# A binary MDR (multidrug-resistant) label is created based on the "total_resistance_classes" feature, where isolates with a resistance breadth above the median are labeled as 1 (MDR) and those below or equal to the median are labeled as 0 (non-MDR).
df["MDR_label"] = (df["total_resistance_classes"] > df["total_resistance_classes"].median()).astype(int)

# Optional metadata features
X_meta = df[["Genome_Length_BP", "GC_Content_Percent"]].copy()

# Final ML table (no leakage)
out = pd.concat(
    [df[["Isolate_ID", "total_resistance_classes", "MDR_label"]], X_meta, X_gene],
    axis=1
)

# The final feature table, which includes the isolate ID, resistance breadth, MDR label, optional metadata features, and gene presence/absence features, is saved to a CSV file for use in machine learning modeling.
out.to_csv(OUT, index=False)
print("Saved:", OUT)
print("Shape:", out.shape)
print(out["MDR_label"].value_counts())

"""
I developed this script to learn how raw biological annotations can be converted into machine learning features in a reproducible way.
The script extracts gene presence/absence information, generates resistance breadth metrics, constructs a binary MDR label for classification, 
and produces a clean feature table for modeling. 
The goal of this step was to understand feature engineering workflows commonly used in bioinformatics 
and machine learning applications.

"""