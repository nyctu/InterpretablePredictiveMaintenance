# Interpretable Predictive Maintenance: Combining Anomaly Detection with Quantitative Root Cause Analysis

This repository contains the code and experimental setup for the paper:

"Interpretable Predictive Maintenance: Combining Anomaly Detection with Quantitative Root Cause Analysis"  
_Submitted to EPIA 2025_

## Overview

This study presents a data-driven, interpretable framework that bridges Anomaly Detection and Root Cause Analysis in Predictive Maintenance. We propose the use of Long Short-Term Memory (LSTM) networks trained with explicit degradation stage labels to detect transition phases in multivariate time series.

Following detection, we apply a two-stage Root Cause Analysis:

1. Clustering of degradation transitions to identify distinct failure modes and track their temporal evolution.
2. SHAP-based sequential pattern mining on vehicle-level data to group vehicles with similar degradation patterns and uncover consistent feature attributions. In particular, early failure symptoms are identified.

The methodology is validated on real-world industrial data from Component X in the SCANIA Industrial Challenge 2024.

## Data Access

Due to file size constraints, the dataset used in this project is not included in the repository.

To run the notebooks, you must manually download the data from the official source:

Download link: https://researchdata.se/sv/catalogue/dataset/2024-34/2  
DOI: https://doi.org/10.5878/jvb5-d390

After downloading, place all data files directly in the root folder of the repository (at the same level as the notebooks), so that the file structure looks like this:

train_operational_readouts.csv  
train_specifications.csv  
train_tte.csv  
validation_operational_readouts.csv  
validation_specifications.csv  
validation_labels.csv  
test_operational_readouts.csv  
test_specifications.csv  
test_labels.csv

Once the files are placed correctly, the notebooks should run without any further configuration.

## Repository Structure

EDA and Preprocessing.ipynb  
Modeling (AD and RCA).ipynb  
documentation/  
├── 2024_IDA_challenge_v2.pdf  
└── Scania_component_X_PdM.pdf

## Important Note on Notebook Execution Order

The three Jupyter notebooks in this repository are designed to be executed in sequence, as they progressively generate and rely on intermediate data files. Running them out of order may result in missing files or inconsistent results.

Recommended execution order:

1. EDA and Preprocessing.ipynb  
   Performs initial exploration and prepares the raw data.

2. Additional Preprocessing.ipynb  
   Applies supplementary transformations and feature engineering required for modeling.

3. Modeling (AD and RCA).ipynb  
   Trains models, performs anomaly detection, and executes root cause analysis.

Please ensure that each notebook completes successfully before proceeding to the next.

## Key Features

- LSTM-based degradation detection with interpretable outputs  
- Two-stage root cause analysis: clustering transitions and SHAP-based sequential pattern mining  
- Modular implementation using Jupyter Notebooks for transparency and reproducibility  
- Uses real industrial data from a complex multi-sensor system  
- Domain-agnostic and adaptable to settings with limited root cause labels
