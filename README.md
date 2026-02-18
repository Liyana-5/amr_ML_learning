# Machine Learning on Antimicrobial Resistance (AMR) Dataset

## Overview

This repository contains notebooks and scripts developed as part of my learning process in applying machine learning methods to biological data. The goal of this project is to understand how biological datasets can be transformed into machine learning–ready feature matrices, how models are trained and evaluated, and how computational results can be interpreted in a biological context.

The analyses are performed using an Antimicrobial Resistance (AMR) dataset obtained from Kaggle , containing annotated bacterial genomes with resistance gene information and resistance phenotype summaries.

This repository is intended for learning and educational purposes, focusing on understanding machine learning workflows rather than developing a production-ready predictive model.

## Why This Dataset

This dataset was selected because:

1) The biological meaning of features is clear (resistance genes → resistance phenotypes)

2) Features are already structured for machine learning exploration

3) It allows learning both computational and biological interpretation together

4) The relationship between genotype and phenotype is easier to understand for beginners

Using an AMR dataset made it possible to learn:

1) Feature engineering from biological annotations

2) Binary classification workflows

3) Model benchmarking across algorithms

4) Interpretation of model outputs in a biological context

## Machin Learning Workflow

The following concepts were explored in this repository:

- Feature engineering from gene annotations

- Creation of a binary resistance label for classification

- Train–test splitting and stratified cross-validation

- Model benchmarking (Random Forest, Logistic Regression, Gradient Boosting)

- Evaluation using ROC-AUC, Precision–Recall curves, and F1 score

- Feature importance analysis (Random Forest importance and permutation importance)

- Biological interpretation of predictive features

##  Model Performance
The models achieved very high performance (ROC-AUC and PR-AUC close to 1.0). This is expected in this dataset because:

- Resistance genes used as features are directly related to resistance outcomes

- The dataset is small and relatively homogeneous

- Strong patterns already exist between gene presence and resistance breadth

Therefore, the high performance reflects strong existing signal in the dataset rather than evidence of a highly generalizable predictive model.

The purpose of these results is to understand evaluation metrics and model behaviour rather than to claim predictive superiority.

# Reference

@dataset{kulkarni_amr_dataset_2024,
  title={AMR Genome Dataset: Antimicrobial Resistance Prediction Dataset},
  author={Kulkarni, Vihaan},
  year={2024},
  publisher={GitHub},
  url={https://github.com/vihaankulkarni29/amr-dataset}
}