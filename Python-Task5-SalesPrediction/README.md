# Sales Prediction Using Python

**Program:** Oasis Infobyte SIP — AICTE OIBSIP
**Track:** Data Science
**Task:** Task 5 — Sales Prediction Using Python

## Objective

Build a regression model that predicts product sales based on advertising spend across
TV, Radio, and Newspaper channels.

## Dataset

The classic [Advertising dataset](https://www.statlearning.com/) from *An Introduction to
Statistical Learning* — 200 markets, each with TV/Radio/Newspaper advertising budgets
(in $ thousands) and resulting product sales (in thousand units).

## Tech Stack

- Python 3, pandas, numpy
- scikit-learn (`LinearRegression`, `RandomForestRegressor`, metrics)
- matplotlib, seaborn
- Jupyter Notebook

## Pipeline

1. **Load & inspect** — no nulls or duplicates in this dataset.
2. **EDA** — pairplot, per-channel scatter plots against Sales, correlation heatmap.
3. **Train/test split** — 80/20.
4. **Model training** — Linear Regression (baseline, interpretable coefficients) + Random
   Forest Regressor (checks for non-linearity).
5. **Evaluation** — MAE, RMSE, R² for both models.
6. **Residual analysis** — checks the best model's errors are random noise, not a
   systematic pattern.
7. **Channel impact interpretation** — the key finding (see below).

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 1.461 | 1.782 | 0.899 |
| Random Forest Regressor | 0.643 | 0.781 | **0.981** |

## Key Finding: which channel matters most?

This is the interesting part, and the notebook doesn't oversimplify it — **the two models
answer two different questions**:

- **Random Forest feature importance** ranks **TV** far above Radio (0.63 vs 0.36) — TV
  explains the most overall variation in sales, because campaigns spend far more on it
  in absolute terms ($0.7k–$296.4k range) and it correlates strongly with sales (r ≈ 0.78).
- **Linear Regression coefficients** show the opposite ranking for *marginal* return:
  **Radio** has the larger per-dollar effect (0.189 vs TV's 0.045) — an incremental radio
  dollar moves sales more than an incremental TV dollar.
- **Newspaper** loses on both measures (coefficient ≈ 0.003, importance ≈ 1%) — the
  easiest line item to cut.

**Practical takeaway:** "which channel drives current sales the most" (TV) and "which
channel gives the best return on the next dollar" (Radio) are different, both valid
questions — a real budget-reallocation recommendation should lean on the second.

## Key Visuals

- `screenshots/01_pairplot.png`
- `screenshots/02_sales_vs_channel_scatter.png`
- `screenshots/03_correlation_heatmap.png`
- `screenshots/04_residuals_and_actual_vs_predicted.png`
- `screenshots/05_channel_impact_comparison.png` — the coefficient-vs-importance contrast

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook Sales_Prediction.ipynb
```
(Or upload the notebook + `advertising.csv` to Google Colab and Run All.)

## Files

```
OIBSIP_Python_Task5_SalesPrediction/
├── Sales_Prediction.ipynb   # Full notebook, executed with outputs
├── advertising.csv          # Dataset
├── README.md                # This file
└── screenshots/             # Exported chart images
```

## Author

*Submitted as part of the AICTE OIBSIP Python Programming track, Task 5.*
