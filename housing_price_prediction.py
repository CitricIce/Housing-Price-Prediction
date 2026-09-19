"""
Housing Price Prediction using Regression Models
==================================================
Loads housing data, explores it, engineers features, and compares three
regression models (Linear Regression, Ridge, and Random Forest) to predict
sale price.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("data/housing_data.csv")
print("Dataset shape:", df.shape)
print(df.describe().round(1))

# ---------------------------------------------------------------------------
# 2. Exploratory data analysis
# ---------------------------------------------------------------------------
corr = df.corr(numeric_only=True)["sale_price"].sort_values(ascending=False)
print("\nCorrelation with sale_price:\n", corr)

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
axes[0, 0].scatter(df["sqft"], df["sale_price"], alpha=0.3, s=10)
axes[0, 0].set_xlabel("Square footage")
axes[0, 0].set_ylabel("Sale price")
axes[0, 0].set_title("Price vs. square footage")

axes[0, 1].scatter(df["age_years"], df["sale_price"], alpha=0.3, s=10, color="darkorange")
axes[0, 1].set_xlabel("Age (years)")
axes[0, 1].set_ylabel("Sale price")
axes[0, 1].set_title("Price vs. age")

df.boxplot(column="sale_price", by="neighborhood_quality", ax=axes[1, 0])
axes[1, 0].set_title("Price by neighborhood quality")
axes[1, 0].set_xlabel("Neighborhood quality (1-5)")
plt.suptitle("")

axes[1, 1].hist(df["sale_price"], bins=40, color="seagreen")
axes[1, 1].set_title("Sale price distribution")
axes[1, 1].set_xlabel("Sale price")

plt.tight_layout()
plt.savefig("plots/eda_overview.png", dpi=120)
plt.close()
print("\nSaved EDA plots -> plots/eda_overview.png")

# ---------------------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------------------
X = df.drop(columns="sale_price")
y = df["sale_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 4. Train models
# ---------------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=10.0),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42),
}

results = []
for name, model in models.items():
    if name == "Random Forest":
        # Tree models don't need scaling
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
print("\nModel comparison:\n", results_df.round(3).to_string(index=False))
results_df.to_csv("model_comparison.csv", index=False)

# ---------------------------------------------------------------------------
# 5. Feature importance (Random Forest)
# ---------------------------------------------------------------------------
rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nRandom Forest feature importances:\n", importances.round(3))

plt.figure(figsize=(8, 5))
importances.plot(kind="barh", color="steelblue")
plt.gca().invert_yaxis()
plt.title("Feature importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=120)
plt.close()
print("Saved feature importance plot -> plots/feature_importance.png")
