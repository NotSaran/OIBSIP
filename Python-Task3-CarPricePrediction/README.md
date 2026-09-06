# Car Price Prediction with Machine Learning

**Program:** Oasis Infobyte SIP — AICTE OIBSIP
**Track:** Data Science
**Task:** Task 3 — Car Price Prediction with Machine Learning

## Objective

Build a regression model that predicts the selling price of a used car based on features
such as model, age, mileage, fuel type, and transmission.

## Dataset

[Vehicle dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
(classic 301-row `car data.csv` version) — used-car listings with model name, year,
selling price, original showroom price, kilometers driven, fuel type, seller type,
transmission, and ownership history.

## Tech Stack

- Python 3, pandas, numpy
- scikit-learn (`LinearRegression`, `RandomForestRegressor`, metrics)
- matplotlib, seaborn
- Jupyter Notebook

## Pipeline

1. **Load & clean** — check shape, nulls, duplicates; standardize categorical text casing.
2. **Feature engineering** — derive `Car_Age` from `Year`; drop the high-cardinality
   `Car_Name` field (98 unique values across only ~300 rows would cause one-hot encoding
   to overfit) and let `Present_Price` carry brand/trim-tier signal instead.
3. **EDA** — price distribution, price by fuel type, price vs. age scatter.
4. **Encoding** — one-hot encode `Fuel_Type`, `Seller_Type`, `Transmission`.
5. **Correlation analysis** — heatmap of numeric features against `Selling_Price`.
6. **Train/test split** — 80/20.
7. **Model training** — Linear Regression and Random Forest Regressor, trained side by side.
8. **Evaluation** — MAE, RMSE, R² for each model; actual-vs-predicted scatter plots.
9. **Feature importance** — from the Random Forest model.
10. **Honest interpretation** — see Results below.

## Results

| Model | MAE (lakh INR) | RMSE (lakh INR) | R² |
|---|---|---|---|
| Linear Regression | 1.47 | 2.52 | **0.753** |
| Random Forest Regressor | 1.49 | 3.56 | 0.508 |

**Linear Regression outperformed Random Forest here** — a genuinely useful result to
understand rather than a "wrong" outcome. With only ~240 training rows and a target driven
overwhelmingly by one strongly linear relationship (`Present_Price`, r = 0.88 with
`Selling_Price`), there isn't enough data or non-linearity for an untuned Random Forest to
earn its extra complexity. The notebook discusses this directly instead of picking whichever
model "should" win by convention.

**Feature importance** (Random Forest): `Present_Price` dominates at ~90% importance,
followed by `Car_Age` (~7%) — confirming that a car's resale value is fundamentally
anchored to its original price, with age as the main secondary adjustment.

## Key Visuals

- `screenshots/01_price_distribution_and_fuel_type.png`
- `screenshots/02_price_vs_age_scatter.png` — depreciation curve
- `screenshots/03_correlation_heatmap.png`
- `screenshots/04_actual_vs_predicted.png` — both models compared
- `screenshots/05_feature_importance.png`

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook Car_Price_Prediction.ipynb
```
(Or upload the notebook + `car_data.csv` to Google Colab and Run All.)

## Files

```
OIBSIP_Python_Task3_CarPricePrediction/
├── Car_Price_Prediction.ipynb   # Full notebook, executed with outputs
├── car_data.csv                 # Dataset
├── README.md                    # This file
└── screenshots/                 # Exported chart images
```

## Author

*Submitted as part of the AICTE OIBSIP Python Programming track, Task 3.*
