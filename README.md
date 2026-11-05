# Advanced Time-Series Forecasting with LSTM and Fourier Features

## Overview

This project implements an advanced time-series forecasting pipeline that combines **Long Short-Term Memory (LSTM) neural networks** with **Fourier-based seasonal feature engineering**.

The objective is to investigate whether explicitly incorporating periodic information through Fourier harmonics can improve the forecasting performance of recurrent neural networks.

The pipeline integrates trend-residual decomposition, temporal feature engineering, multi-seed experiments, Fourier harmonic ablation, rolling-origin validation, baseline comparisons, and final evaluation on an untouched test set.

The implementation is built with PyTorch and uses a structured experimental framework to investigate the contribution of seasonal representations to neural time-series forecasting.

## Key Features

- **Logarithmic Transformation:** Stabilizes the target series through logarithmic scaling.
- **Trend-Residual Decomposition:** Separates the estimated linear trend from the residual component.
- **Fourier Feature Engineering:** Encodes periodic seasonal patterns using sine and cosine functions.
- **Stacked LSTM Architecture:** Learns temporal dependencies through multiple recurrent layers.
- **Fourier Harmonic Ablation:** Investigates the effect of different numbers of seasonal harmonics.
- **Multi-Seed Experiments:** Evaluates model stability across different random initializations.
- **Rolling-Origin Validation:** Tests forecasting consistency across multiple chronological validation windows.
- **Baseline Comparisons:** Evaluates the proposed architecture against Seasonal Naive, Fourier Regression, and Vanilla LSTM.
- **Residual Diagnostics:** Examines prediction errors and their autocorrelation.
- **Model Checkpointing:** Saves trained weights, model configuration, and preprocessing parameters.

## Methodology

### 1. Data Preparation

The pipeline uses monthly alcohol sales data.

| Parameter | Configuration |
|---|---|
| Dataset | Alcohol Sales |
| Date column | `observation_date` |
| Target column | `S4248SM144NCEN` |
| Target variable | Alcohol sales |
| Sampling frequency | Monthly |
| Transformation | Natural logarithm |

The preprocessing stage includes:

- Converting dates into datetime format.
- Converting target values into numeric format.
- Removing missing observations.
- Sorting observations chronologically.
- Removing duplicate dates.
- Checking for non-positive target values.

The target series is transformed using:

$$
y_t^{\log} = \log(y_t)
$$

### 2. Temporal Data Splitting

The dataset is divided chronologically to preserve temporal ordering and avoid random train-test splitting.

| Dataset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

The training set is used for parameter estimation and model optimization.

The validation set is used for hyperparameter selection and early stopping.

The test set remains untouched during model selection and is reserved for final performance evaluation.

### 3. Trend and Residual Decomposition

The logarithmically transformed target is decomposed into a linear trend and a residual component.

$$
\log(y_t) = T_t + R_t
$$

Where:

- $T_t$: Estimated linear trend.
- $R_t$: Residual component.

A linear trend is estimated using polynomial regression of degree one, fitted exclusively on the training observations.

The residual component is calculated as:

$$
R_t = \log(y_t) - T_t
$$

The residuals are standardized using `StandardScaler`, fitted on the training data.

The estimated trend is normalized using its training-derived minimum and maximum values.

This decomposition allows the LSTM to focus on temporal fluctuations while retaining the underlying trend information as an additional feature.

### 4. Fourier Feature Engineering

Fourier features are introduced to represent recurring seasonal patterns.

For each harmonic $k$, two periodic features are generated:

**Sine component**

$$
S_k(t) = \sin\left(\frac{2\pi kt}{P}\right)
$$

**Cosine component**

$$
C_k(t) = \cos\left(\frac{2\pi kt}{P}\right)
$$

Where:

- $t$: Temporal index.
- $k$: Harmonic number.
- $P$: Seasonal period, set to 12 months.
- $K$: Total number of Fourier harmonics.

The number of harmonics is investigated using:

$$
K \in \{0,1,2,3,4,5\}
$$

Each harmonic contributes two additional input features.

A configuration with $K=0$ represents a model without Fourier features and serves as the Vanilla LSTM configuration.

### 5. Feature Matrix Construction

The final feature matrix combines three components:

1. Standardized residuals.
2. Fourier sine and cosine features.
3. Normalized linear trend.

The resulting feature vector at each time step is:

$$
X_t =
\left[
\widetilde{R}_t,
S_1(t),C_1(t),
\ldots,
S_K(t),C_K(t),
\widetilde{T}_t
\right]
$$

Where:

- $\widetilde{R}_t$: Standardized residual.
- $\widetilde{T}_t$: Normalized trend.
- $S_k(t)$ and $C_k(t)$: Fourier features.

The total input dimensionality is:

$$
D = 2K + 2
$$

### 6. Sequence Generation

A sliding-window approach converts the feature matrix into temporal sequences.

The model uses a lookback window of 36 observations to predict the residual component at the next time step.

For each prediction:

$$
X_t =
[X_{t-36},\ldots,X_{t-1}]
$$

The corresponding target is:

$$
\hat{R}_t = f(X_t)
$$

This allows the network to learn temporal dependencies across the previous three years of observations.

### 7. LSTM Architecture

The forecasting model consists of two stacked LSTM layers followed by a fully connected regression head.

**Architecture configuration:**

| Component | Configuration |
|---|---|
| Input | Sequential multivariate features |
| LSTM layers | 2 |
| Hidden dimension | 64 |
| LSTM dropout | 0.15 |
| Fully connected layer | 64 → 32 |
| Activation | GELU |
| Dropout | 0.15 |
| Output layer | 32 → 1 |
| Prediction target | Standardized residual |

The final hidden representation is extracted from the last time step and passed through the regression head to produce the next-step residual prediction.

### 8. Training Configuration

The model is optimized using AdamW with mean squared error as the training objective.

| Hyperparameter | Value |
|---|---:|
| Lookback window | 36 |
| Hidden size | 64 |
| Number of LSTM layers | 2 |
| Dropout | 0.15 |
| Batch size | 32 |
| Initial learning rate | 0.0005 |
| Weight decay | 0.00001 |
| Maximum epochs | 150 |
| Early stopping patience | 25 |
| Gradient clipping | 1.0 |
| Optimizer | AdamW |
| Loss function | MSE |
| LR scheduler | ReduceLROnPlateau |

The learning rate scheduler reduces the learning rate by a factor of 0.5 when validation loss stops improving, with a minimum learning rate of 0.00001.

Gradient clipping is applied to stabilize optimization.

Early stopping monitors validation loss and restores the model weights corresponding to the best validation performance.

## Experimental Design

The experimental framework consists of five main stages.

### Experiment 1 — Baseline Evaluation

Two conventional forecasting approaches are implemented.

**Seasonal Naive**

Uses the observation from the same month of the previous year:

$$
\hat{y}_t = y_{t-12}
$$

**Fourier Linear Regression**

Uses a linear regression model with a temporal trend and Fourier seasonal features:

$$
\hat{z}_t =
\beta_0 + \beta_1 t +
\sum_{k=1}^{K}
\left[
a_k S_k(t) + b_k C_k(t)
\right]
$$

Where $z_t = \log(y_t)$.

The Fourier regression baseline is evaluated using different harmonic configurations:

$$
K \in \{0,1,2,3,4,5\}
$$

### Experiment 2 — Fourier Harmonic Ablation

The LSTM architecture is trained using different Fourier configurations.

$$
K \in \{0,1,2,3,4,5\}
$$

Each configuration is evaluated using three random seeds:

$$
\text{Seeds} = \{42,123,2026\}
$$

This produces:

$$
6 \times 3 = 18
$$

independent training experiments.

For each experiment, the following metrics are recorded:

- Validation RMSE.
- Validation MAE.
- Validation MAPE.
- Best validation epoch.
- Best validation loss.

The results are aggregated across random seeds using mean and standard deviation.

The optimal harmonic configuration is selected based on the lowest mean validation RMSE.

### Experiment 3 — Rolling-Origin Validation

Rolling-origin validation is performed to investigate the temporal stability of the selected architecture.

The selected Fourier configuration is evaluated across three chronological folds.

The training window expands at each fold, while the validation horizon is set to 12 observations.

The rolling origins are selected from approximately 60% to 80% of the development dataset.

For each fold:

1. A new preprocessing pipeline is fitted using the available training observations.
2. A new LSTM model is initialized.
3. The model is trained using the selected Fourier configuration.
4. Early stopping determines the best training epoch.
5. Predictions are generated for the corresponding validation horizon.
6. RMSE, MAE, and MAPE are calculated.

This experiment provides additional insight into forecasting performance across different temporal periods.

### Experiment 4 — Final Model Training

After selecting the optimal number of Fourier harmonics, the final model is retrained using the combined training and validation data.

The final training epoch count is determined using the median best epoch across the three selected-hyperparameter experiments.

The final model is trained with a fixed number of epochs rather than using the test set for early stopping.

### Experiment 5 — Final Test Evaluation

The final hybrid LSTM + Fourier model is evaluated on the untouched test set.

Its performance is compared against:

- Seasonal Naive.
- Fourier Linear Regression.
- Vanilla LSTM.
- Hybrid LSTM + Fourier.

All final comparisons use the original sales scale rather than the standardized residual scale.

## Evaluation Metrics

Three regression metrics are used to evaluate forecasting performance.

**Root Mean Squared Error (RMSE)**

$$
RMSE =
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
}
$$

RMSE penalizes larger prediction errors more heavily.

**Mean Absolute Error (MAE)**

$$
MAE =
\frac{1}{N}
\sum_{i=1}^{N}
|y_i-\hat{y}_i|
$$

MAE measures the average absolute prediction error.

**Mean Absolute Percentage Error (MAPE)**

$$
MAPE =
\frac{100}{N}
\sum_{i=1}^{N}
\left|
\frac{y_i-\hat{y}_i}{y_i}
\right|
$$

MAPE expresses prediction errors as percentages of actual observations.

Lower values indicate better forecasting performance across all three metrics.

## Visualization and Diagnostics

The notebook generates interactive visualizations using Plotly.

The visual analysis includes:

- Original monthly alcohol sales.
- Log-transformed sales.
- Estimated linear trend.
- Actual versus predicted test observations.
- Final model training loss.
- Fourier harmonic ablation results.
- Prediction errors over time.
- Prediction error distribution.
- Autocorrelation of test prediction errors.

These visualizations help evaluate model convergence, forecasting accuracy, seasonal behavior, and remaining temporal structure in prediction errors.

## Installation and Requirements

The project is implemented in Python using the following libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
pip install torch plotly tqdm jupyter
```

Google Colab is also supported.

## Dataset Setup

The notebook expects a CSV file named:

```text
Alcohol sales.csv
```

The dataset must contain the following columns:

```text
observation_date
S4248SM144NCEN
```

The notebook uses Google Colab's file upload interface to load the dataset.

## Running the Project

1. Clone or download the repository.
2. Install the required dependencies.
3. Download the alcohol sales dataset.
4. Place `Alcohol sales.csv` in the working directory or upload it when prompted.
5. Open `LSTMTASK.ipynb` in Jupyter Notebook or Google Colab.
6. Execute the notebook cells sequentially.
7. Review the experimental results, evaluation metrics, and generated visualizations.

## Output Files

The pipeline exports experimental results and the final trained model.

| File | Description |
|---|---|
| `best_lstm_fourier_v2.pth` | Final model checkpoint and preprocessing parameters |
| `lstm_fourier_experiments.csv` | Individual Fourier ablation results |
| `fourier_ablation_summary.csv` | Aggregated harmonic ablation statistics |
| `baseline_validation_test.csv` | Baseline validation and test results |
| `final_model_comparison.csv` | Final test performance comparison |
| `rolling_validation_results.csv` | Rolling-origin validation results |

The rolling validation CSV is generated when rolling-origin validation is enabled and produces valid folds.

## Reproducibility

The implementation includes random seed control for Python, NumPy, and PyTorch.

The experiments use the following random seeds:

$$
\{42,123,2026\}
$$

PyTorch deterministic algorithms are enabled where supported, and cuDNN benchmarking is disabled.

The final checkpoint stores:

- Model state dictionary.
- Model architecture configuration.
- Selected Fourier harmonic count.
- Fourier seasonal period.
- Lookback window.
- Trend coefficients.
- Trend normalization parameters.
- Residual scaler parameters.
- Final training epoch count.
- Random seed.
- Feature ordering.
- Final test metrics.

This allows the trained model and its preprocessing configuration to be restored for subsequent inference.

## Research Objective

The central research question is:

**Can explicitly incorporating Fourier-based seasonal representations improve the forecasting performance of LSTM networks?**

By systematically evaluating different harmonic configurations, comparing against conventional forecasting baselines, and investigating performance across multiple validation windows, this project examines the contribution of explicit seasonal feature engineering to neural time-series forecasting.

## Technologies

**Python · PyTorch · NumPy · Pandas · Scikit-learn · Plotly · Deep Learning · LSTM · Fourier Analysis · Time-Series Forecasting**
