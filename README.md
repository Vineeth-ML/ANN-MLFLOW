# ANN-MLFLOW

A deep learning project that trains an Artificial Neural Network (ANN) to predict white wine quality scores, using **Hyperopt** for automated hyperparameter tuning and **MLflow** for experiment tracking and model management.

---

## Overview

This project builds a regression model on the [UCI Wine Quality dataset](https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv) to predict wine quality on a scale based on physicochemical properties. The workflow covers data preparation, model training, hyperparameter optimization, and experiment logging — all in a single Jupyter notebook.

---

## Features

- **ANN with Keras** — Sequential model with input normalization and ReLU activations
- **Hyperparameter tuning with Hyperopt** — Bayesian optimization (TPE) over learning rate and SGD momentum
- **Experiment tracking with MLflow** — Logs parameters, metrics, and model artifacts for every trial
- **Best model selection** — Automatically identifies and saves the run with the lowest RMSE

---

## Project Structure

```
ANN-MLFLOW/
│
├── ANN.ipynb          # Main notebook: data prep, training, tuning, logging
└── README.md
```

---

## Model Architecture

```
Input (11 features)
  → Normalization layer (mean/variance from training data)
  → Dense(64, activation='relu')
  → Dense(1)  ← predicted quality score
```

**Loss:** Mean Squared Error  
**Optimizer:** SGD with tunable learning rate and momentum  
**Metric:** Root Mean Squared Error (RMSE)

---

## Hyperparameter Search Space

| Parameter    | Distribution              | Range         |
|--------------|---------------------------|---------------|
| `lr`         | Log-uniform               | 1e-5 → 1e-1   |
| `momentum`   | Uniform                   | 0.0 → 1.0     |

Optimization runs for **4 evaluations** using the Tree-structured Parzen Estimator (TPE) algorithm.

---


## Dataset

- **Source:** [UCI Wine Quality (White)](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Samples:** ~4,898 white wine observations
- **Features:** 11 physicochemical inputs (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free/total sulfur dioxide, density, pH, sulphates, alcohol)
- **Target:** Quality score (integer, 0–10)
- **Split:** 80% train / 20% test; 2% of train reserved for validation

---

## Results

After tuning, the best model parameters and RMSE are printed at the end of the notebook and logged as the final MLflow run. You can compare all runs in the MLflow tracking UI under the **Wine-quality** experiment.

---
