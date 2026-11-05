# Advanced Time-Series Forecasting with LSTM and Fourier Features

## Overview

This project implements an advanced time-series forecasting pipeline that combines **Long Short-Term Memory (LSTM) networks** with **Fourier-based seasonal features** to model temporal dependencies and recurring seasonal patterns.

The primary objective is to investigate whether incorporating Fourier harmonics into an LSTM architecture improves forecasting performance compared with conventional forecasting approaches.

The implementation uses a structured experimental framework involving multiple random seeds, Fourier feature ablation, rolling-origin validation, baseline comparisons, and final evaluation on an untouched test set.

## Key Features

- **Logarithmic Transformation:** Applies a logarithmic transformation to stabilize the target series.
- **Trend and Residual Decomposition:** Separates the linear trend from the residual component.
- **Fourier Feature Engineering:** Generates sinusoidal and cosine features to capture periodic seasonal patterns.
- **LSTM Architecture:** Uses a multi-layer LSTM network to learn temporal dependencies.
- **Hyperparameter Ablation:** Evaluates different numbers of Fourier harmonics.
- **Multi-Seed Experiments:** Repeats experiments across multiple random seeds to evaluate stability.
- **Rolling-Origin Validation:** Evaluates model performance across multiple chronological validation windows.
- **Baseline Comparisons:** Compares the proposed model against Seasonal Naive, Fourier Regression, and Vanilla LSTM.
- **Residual Diagnostics:** Analyzes prediction errors and their autocorrelation.
- **Model Checkpointing:** Saves the trained model and preprocessing parameters for future use.

## Methodology

### 1. Data Preparation

The pipeline loads monthly alcohol sales data from a CSV file.

| Parameter          | Configuration      |
| ------------------ | ------------------ |
| Date column        | `observation_date` |
| Target column      | `S4248SM144NCEN`   |
| Target variable    | Alcohol sales      |
| Sampling frequency | Monthly            |
| Transformation     | Natural logarithm  |

The dataset is sorted chronologically, missing observations are removed, duplicate dates are eliminated, and non-positive target values are rejected.

### 2. Temporal Data Splitting

The dataset is divided chronologically to prevent future observations from leaking into model training.

| Dataset    | Percentage |
| ---------- | ---------: |
| Training   |        70% |
| Validation |        15% |
| Testing    |        15% |

The test set remains untouched during model selection and hyperparameter experiments.

### 3. Trend and Residual Decomposition

The logarithmically transformed target is decomposed into a linear trend and a residual component:

[
\log(y_t)=T_t+R_t
]

Where:

- (T_t): Estimated linear trend.
- (R_t): Residual component.

A linear regression trend is fitted using training observations. The residuals are standardized using `StandardScaler`, while the trend is normalized using training-derived minimum and maximum values.

### 4. Fourier Feature Engineering

Fourier features are introduced to represent recurring seasonal patterns.

For each harmonic (k):

[
S_k(t)=\sin\left(\frac{2\pi kt}{P}\right)
]

[
C_k(t)=\cos\left(\frac{2\pi kt}{P}\right)
]

Where:

- (P=12): Seasonal period corresponding to 12 months.
- (K): Number of Fourier harmonics.

The experiment evaluates:

[
K\in{0,1,2,3,4,5}
]

Each harmonic contributes two additional input features.

### 5. Feature Construction

The final input feature matrix combines:

1. Standardized residuals.
2. Fourier sine and cosine components.
3. Normalized linear trend.

The resulting feature vectors are organized into sliding temporal sequences.

The lookback window is set to 36 observations, allowing the model to use the previous 36 months to predict the next observation.

### 6. LSTM Architecture

The forecasting model consists of:

- Two stacked LSTM layers.
- Hidden dimension of 64.
- Dropout regularization.
- A fully connected regression head.
- GELU activation.
- A single output neuron.

**Model configuration:**

| Hyperparameter          |   Value |
| ----------------------- | ------: |
| Lookback window         |      36 |
| Hidden size             |      64 |
| LSTM layers             |       2 |
| Dropout                 |    0.15 |
| Batch size              |      32 |
| Learning rate           |  0.0005 |
| Weight decay            | 0.00001 |
| Maximum epochs          |     150 |
| Early stopping patience |      25 |
| Optimizer               |   AdamW |
| Loss function           |     MSE |

### 7. Experimental Design

The project contains several experiments.

**Experiment 1 — Baseline Evaluation**

Evaluates:

- Seasonal Naive forecasting.
- Fourier Linear Regression.

**Experiment 2 — Fourier Ablation**

Trains the LSTM architecture using different Fourier configurations:

[
K\in{0,1,2,3,4,5}
]

Each configuration is evaluated using three random seeds:

[
\text{Seeds}={42,123,2026}
]

The optimal number of harmonics is selected based on the lowest mean validation RMSE.

**Experiment 3 — Rolling-Origin Validation**

Performs three chronological validation folds to investigate the consistency of the selected architecture across different forecasting origins.

Each fold uses an expanding training window and a 12-observation validation horizon.

**Experiment 4 — Final Model Training**

After selecting the optimal Fourier configuration, the final model is retrained using the combined training and validation data.

The final training epoch count is determined using the median best epoch from the selected validation experiments.

**Experiment 5 — Final Test Evaluation**

The final model is evaluated on the previously untouched test set.

Its performance is compared against:

- Seasonal Naive.
- Fourier Regression.
- Vanilla LSTM.
- Hybrid LSTM + Fourier.

## Evaluation Metrics

The implementation uses three regression metrics.

**Root Mean Squared Error (RMSE)**

[
RMSE=\sqrt{\frac{1}{N}\sum\_{i=1}^{N}(y_i-\hat y_i)^2}
]

**Mean Absolute Error (MAE)**

[
MAE=\frac{1}{N}\sum\_{i=1}^{N}|y_i-\hat y_i|
]

**Mean Absolute Percentage Error (MAPE)**

[
MAPE=\frac{100}{N}\sum\_{i=1}^{N}
\left|\frac{y_i-\hat y_i}{y_i}\right|
]

Lower values indicate better forecasting performance.

## Visualization and Diagnostics

The notebook generates several visualizations:

- Original sales and logarithmically transformed sales.
- Estimated linear trend.
- Actual versus predicted test observations.
- Training loss across epochs.
- Fourier harmonic ablation results.
- Prediction error over time.
- Prediction error distribution.
- Autocorrelation of test prediction errors.

These visualizations help assess forecasting accuracy, model behavior, and remaining temporal structure in prediction errors.

## Installation and Requirements

The project is implemented in Python using the following libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
pip install torch plotly tqdm jupyter
```

Google Colab is supported.

## Dataset Setup

Place the dataset in the working directory using the following filename:

```text
Alcohol sales.csv
```

The notebook expects the following columns:

```text
observation_date
S4248SM144NCEN
```

In Google Colab, upload the CSV file when prompted by the notebook.

## Running the Project

1. Clone or download the repository.
2. Install the required dependencies.
3. Download and prepare the dataset.
4. Place `Alcohol sales.csv` in the working directory.
5. Open `LSTMTASK.ipynb` in Jupyter Notebook or Google Colab.
6. Execute the notebook sequentially.
7. Review the experimental results, visualizations, and saved model checkpoint.

## Output Files

The pipeline generates the following artifacts:

| File                             | Description                                         |
| -------------------------------- | --------------------------------------------------- |
| `best_lstm_fourier_v2.pth`       | Final model checkpoint and preprocessing parameters |
| `lstm_fourier_experiments.csv`   | Individual Fourier ablation experiment results      |
| `fourier_ablation_summary.csv`   | Aggregated ablation statistics                      |
| `baseline_validation_test.csv`   | Baseline validation and test results                |
| `final_model_comparison.csv`     | Final test performance comparison                   |
| `rolling_validation_results.csv` | Rolling-origin validation results                   |

## Reproducibility

The implementation includes random seed control for Python, NumPy, and PyTorch.

Experiments are conducted using multiple seeds, and deterministic PyTorch algorithms are enabled where supported.

The final checkpoint stores:

- Model architecture configuration.
- Trained model weights.
- Selected Fourier harmonic count.
- Lookback window.
- Trend coefficients and normalization parameters.
- Residual scaler parameters.
- Final training epoch count.
- Test evaluation metrics.

## Research Objective

The central research question is:

**Can explicitly incorporating Fourier-based seasonal representations improve the forecasting performance and temporal modeling capabilities of LSTM networks?**

By comparing different harmonic configurations against conventional baselines and a Vanilla LSTM, this project investigates the contribution of explicit seasonal feature engineering to neural time-series forecasting.

---

**Technologies:** Python · PyTorch · NumPy · Pandas · Scikit-learn · Plotly · Time-Series Forecasting · Deep Learning · LSTM · Fourier Analysis
