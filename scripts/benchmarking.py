# This script performs benchmarking of different machine learning models on the Kaggle AMR dataset. 
# It uses stratified 5-fold cross-validation to evaluate the performance of Logistic Regression, Random Forest, and Gradient Boosting classifiers. 
# The results are saved to a CSV file and a bar plot comparing the ROC-AUC scores is generated.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
data_path = BASE_DIR / "data" / "Kaggle_AMR_Dataset_v1.0.csv"
# Load the dataset
data = pd.read_csv(data_path)

# import necessary libraries for modeling
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


#data = pd.read_csv("../data/Kaggle_AMR_Dataset_v1.0.csv")

# X = gene_ features only 
X = data[[c for c in data.columns if c.startswith("gene_")]].copy()
y = (data["total_resistance_classes"] > data["total_resistance_classes"].median()).astype(int).to_numpy()

# Define a 5-fold stratified cross-validation strategy.
# The data is split into 5 parts while keeping the class distribution the same in each split,
# allowing more reliable and reproducible model evaluation.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define the models to benchmark. Logistic Regression and Random Forest are set to handle class imbalance with 'balanced' weights, while Gradient Boosting will use sample weights computed from the training data.
models = {
    "LogReg (balanced)": LogisticRegression(max_iter=3000, class_weight="balanced"),
    "RandomForest (balanced)": RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced"),
    "GradBoost": GradientBoostingClassifier(random_state=42),
}

# Function to compute sample weights for handling class imbalance in Gradient Boosting
def compute_sample_weight(y_fold):
    # inverse-frequency weights: minority class gets higher weight
    counts = np.bincount(y_fold)
    w0 = 1.0 / counts[0]
    w1 = 1.0 / counts[1]
    return np.where(y_fold == 1, w1, w0)

# Loop through each model and perform cross-validation, collecting performance metrics (ROC-AUC, PR-AUC, F1 score) for each fold.
rows = []

for name, model in models.items():
    aucs, praucs, f1s = [], [], []
    # For each fold, the model is trained on the training set and evaluated on the test set.
    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        # Clone the model to ensure a fresh instance for each fold, preventing data leakage and ensuring fair evaluation.
        m = clone(model)

        # handle imbalance for models that don't support class_weight
        fit_kwargs = {}
        if name == "GradBoost":
            fit_kwargs["sample_weight"] = compute_sample_weight(y_tr)
       
       # Fit the model on the training data and evaluate its performance on the test data, storing the metrics for later comparison.
        m.fit(X_tr, y_tr, **fit_kwargs)
        # predict probabilities for the positive class and compute metrics
        proba = m.predict_proba(X_te)[:, 1]
        # convert probabilities to binary predictions using a threshold of 0.5 for F1 score calculation
        pred = (proba >= 0.5).astype(int)
        # Append the computed metrics to the respective lists for each fold.
        aucs.append(roc_auc_score(y_te, proba))
        praucs.append(average_precision_score(y_te, proba))
        f1s.append(f1_score(y_te, pred))
    # After all folds are evaluated, the mean and standard deviation of each metric are calculated and stored in a list of dictionaries for later conversion to a DataFrame.
    rows.append({
        "Model": name,
        "ROC-AUC (mean)": np.mean(aucs),
        "ROC-AUC (std)": np.std(aucs),
        "PR-AUC (mean)": np.mean(praucs),
        "PR-AUC (std)": np.std(praucs),
        "F1 (mean)": np.mean(f1s),
        "F1 (std)": np.std(f1s),
    })

# Finally, the collected results are converted into a DataFrame, saved to a CSV file, and a bar plot comparing the ROC-AUC scores of the different models is generated and saved as an image.
results_df = pd.DataFrame(rows).sort_values("ROC-AUC (mean)", ascending=False)
results_df.to_csv(BASE_DIR / "results" / "benchmark_results.csv", index=False)

results_df.set_index("Model")["ROC-AUC (mean)"].plot(kind="bar")
plt.ylabel("ROC-AUC")
plt.title("Model performance comparison")
plt.tight_layout()
plt.savefig(BASE_DIR / "results" / "fig2_model_comparison.png", dpi=300)
plt.show()
