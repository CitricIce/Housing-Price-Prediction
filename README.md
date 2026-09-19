# Housing Price Prediction

A regression project predicting home sale prices from property features (square footage, bedrooms, bathrooms, lot size, age, neighborhood quality, and more).

**Note on the data:** the original inspiration is Kaggle's "House Prices: Advanced Regression Techniques" dataset. Since that dataset requires a Kaggle account to download, this project uses a synthetically generated dataset (`generate_data.py`) built to mirror the same feature set and realistic price relationships (nonlinear age depreciation, neighborhood effects, noise), so the analysis and modeling pipeline below is genuine, even though the underlying rows are simulated.

## What it does

1. **Generates the dataset** (`generate_data.py`) — 1,500 synthetic homes with 10 features and a sale price computed from a realistic (nonlinear) pricing formula plus noise.
2. **Explores the data** (`housing_price_prediction.py`) — correlation analysis and four plots: price vs. square footage, price vs. age, price by neighborhood quality, and the price distribution.
3. **Trains and compares three models**:
   - Linear Regression
   - Ridge Regression
   - Random Forest Regressor
4. **Evaluates** each model with MAE, RMSE, and R².
5. **Reports feature importance** from the Random Forest model.

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Ridge Regression | ~15,300 | ~19,200 | 0.925 |
| Linear Regression | ~15,350 | ~19,250 | 0.924 |
| Random Forest | ~17,950 | ~22,450 | 0.897 |

Square footage dominates the price signal (80% of Random Forest feature importance), followed by neighborhood quality and distance to downtown — consistent with real housing market intuition.

## Files

- `generate_data.py` — builds the synthetic dataset
- `housing_price_prediction.py` — EDA, modeling, and evaluation
- `data/housing_data.csv` — the generated dataset
- `plots/eda_overview.png` — exploratory data analysis plots
- `plots/feature_importance.png` — Random Forest feature importance
- `model_comparison.csv` — model performance metrics

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib
python generate_data.py
python housing_price_prediction.py
```

## Tech

Python, pandas, NumPy, scikit-learn, matplotlib.
