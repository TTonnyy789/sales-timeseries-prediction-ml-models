# Sales Time-Series Prediction with ML Models

A machine learning pipeline for forecasting grocery product sales across multiple stores and product categories. The system segments store-product combinations, selects the best-performing model per segment, and evaluates predictions using Mean Absolute Error (MAE).

## Table of Contents

- [Dataset](#dataset)
- [Pipeline Overview](#pipeline-overview)
- [Models](#models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

---

## Dataset

**`Products_Information.csv`** — Daily sales records for a grocery chain (~250 MB).

| Column | Description |
|---|---|
| `date` | Transaction date |
| `store_nbr` | Store identifier (1–54) |
| `product_type` | Product category (33 categories) |
| `sales` | Units sold |
| `special_offer` | Binary flag for promotional offers |
| `id` | Row identifier |

The dataset covers **January 2013 – August 2017** across **54 stores** and **33 product types**, producing up to ~1,782 unique store-product segments.

---

## Pipeline Overview

### Step 1 — Data Loading & Preprocessing
Load `Products_Information.csv`, parse dates, and check for missing values.

### Step 2 — Exploratory Data Analysis
- Overall sales trends over time
- Sales distribution (histogram)
- Sales by store (boxplot)
- Sales by product type (boxplot)

### Step 3 — Segmentation
Group data by every `(store_nbr, product_type)` pair. Each segment is treated as an independent time series.

Segments are split into two categories:
- **Zero-sales segments**: combinations where total sales = 0 throughout the period (predicted as 0)
- **Non-zero-sales segments**: combinations with at least some sales activity

For non-zero segments, leading zeros (before the first recorded sale) are trimmed so models are not penalised for pre-launch periods.

![Zero vs Non-Zero Sales](pic/portionof%20zero%20and%20non%20zero.png)

### Step 4 — Heatmap Analysis
Average sales per segment are visualised as heatmaps grouped by store range to identify high-volume store-product combinations.

| Group | Stores |
|---|---|
| Group 1 | 1–10 |
| Group 2 | 11–20 |
| Group 3 | 21–30 |
| Group 4 | 31–40 |
| Group 5 | 50–53 |

![Heatmap Group 1](pic/hm1.png)

### Step 5 — Forecasting
For each non-zero segment, five models are trained and evaluated. The model with the lowest MAE on the test set (August 2017) is selected.

**Train/test split:**
- Training: up to 2017-07-31
- Test: 2017-07-31 onwards

---

## Models

All models use **lag features** derived from historical sales and special-offer status. The lag window is tuned per model type.

| Model | Lag Features | Key Hyperparameters |
|---|---|---|
| Linear Regression | 28 sales lags + 28 offer lags | `fit_intercept=True` |
| Random Forest | 14 sales lags + 14 offer lags | `n_estimators=300, max_depth=20` |
| LightGBM | 14 sales lags + 14 offer lags | `n_estimators=400, max_depth=30, lr=0.01` |
| XGBoost | 28 sales lags + 28 offer lags | `n_estimators=200, max_depth=50, lr=0.01` |
| MLP Regressor | 21 sales lags + 21 offer lags | `hidden_layers=(200,200,200,200,100)` |

Hyperparameters were tuned with `GridSearchCV` using `TimeSeriesSplit` cross-validation.

---

## Results

### Model Selection Frequency
The pipeline selects whichever model achieves the lowest MAE per segment.

![Model Selection Frequency](pic/number_of_model.png)

### MAE Distribution
![MAE Distribution](pic/distribution%20of%20MAE.png)

### Average MAE — Best-Model Selection vs Linear-Only Baseline
![Average MAE All Models](pic/average_MAE_allmodels.png)
![Average MAE Linear](pic/average_MAE_linear.png)

### Overall Evaluation
![Overall MAE](pic/Overall_MAE_across5models.png)

---

## Project Structure

```
sales-timeseries-prediction-ml-models/
├── Products_Information.csv       # Raw dataset (required)
├── non_zero_sales_segments.csv    # Pre-filtered non-zero segments
├── zero_sales_segments.csv        # Pre-filtered zero-sales segments
├── results_second_half.csv        # Saved prediction results (second half)
├── forecasters.py                 # SalesForecaster and ZeroSalesForecaster classes
├── main.py                        # Full pipeline: EDA → segmentation → forecasting → evaluation
├── main_2.py                      # Exploratory notebook (ARIMA / RandomForest baseline)
├── requirements.txt
├── Teammates/
│   ├── features(elena).py         # MLP feature experiments
│   └── GradientBoostingRegressor(Scarlett).py
└── pic/                           # Output visualisations
```

---

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies

---

## Installation

```bash
git clone https://github.com/TTonnyy789/sales-timeseries-prediction-ml-models.git
cd sales-timeseries-prediction-ml-models
pip install -r requirements.txt
```

Place `Products_Information.csv` in the project root if it is not already present.

---

## Usage

Run the full pipeline in order by executing each `#%%` cell in `main.py` (works in VS Code, PyCharm, or Jupyter via `jupytext`):

```bash
python main.py
```

Or step through it as a notebook in any IDE that supports `# %%` cell markers.

**Key entry points in `main.py`:**

| Step | What it does |
|---|---|
| Steps 1–2 | Load and clean `Products_Information.csv` |
| Step 3 | EDA plots |
| Steps 4–6 | Segment data; visualise zero vs non-zero splits and heatmaps |
| Step 7-1 | Benchmark: Linear Regression only across all segments |
| Steps 7-2/3 | Best-model selection (first and second halves run separately to distribute compute) |
| Steps 7-4/5 | Load saved results and produce evaluation plots |

**To forecast a specific store-product pair:**

```python
from forecasters import SalesForecaster

SalesForecaster.load_data("Products_Information.csv")

forecaster = SalesForecaster(store_number=1, product_type='BEVERAGES')
model_name, mae, predictions, sales_range, avg_sales, actuals = forecaster.select_and_forecast()
print(f"Best model: {model_name}  |  MAE: {mae:.2f}")
```

---

## License

MIT — see [LICENSE](LICENSE).
