"""
Generates a synthetic but realistic housing dataset for regression practice.
Modeled on the feature set used in Kaggle's "House Prices: Advanced
Regression Techniques" competition (square footage, bedrooms, bathrooms,
lot size, age, neighborhood quality, etc.), with a price formula plus
noise so the relationships are realistic but not perfectly linear.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 1500

sqft = np.random.normal(1800, 650, n).clip(500, 5000)
bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], n, p=[0.05, 0.15, 0.35, 0.30, 0.10, 0.05])
bathrooms = np.round(np.clip(bedrooms * 0.75 + np.random.normal(0, 0.5, n), 1, 5) * 2) / 2
lot_size = np.random.normal(9000, 4000, n).clip(1500, 30000)
age = np.random.exponential(20, n).clip(0, 120).round()
garage_spaces = np.random.choice([0, 1, 2, 3], n, p=[0.1, 0.25, 0.55, 0.10])
neighborhood_quality = np.random.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.2, 0.35, 0.25, 0.10])
distance_to_downtown = np.random.exponential(8, n).clip(0.5, 40)
has_pool = np.random.choice([0, 1], n, p=[0.85, 0.15])
overall_condition = np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.15, 0.45, 0.25, 0.10])

# Price formula: realistic weights + nonlinear age decay + noise
base_price = (
    sqft * 95
    + bedrooms * 4500
    + bathrooms * 6000
    + lot_size * 1.8
    + garage_spaces * 5500
    + neighborhood_quality * 18000
    + overall_condition * 9000
    + has_pool * 12000
    - distance_to_downtown * 1200
    - age * 450 * np.exp(-age / 80)  # depreciation that levels off for older homes
)

noise = np.random.normal(0, 18000, n)
sale_price = (base_price + noise).clip(40000, None).round(-2)

df = pd.DataFrame({
    "sqft": sqft.round().astype(int),
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "lot_size": lot_size.round().astype(int),
    "age_years": age.astype(int),
    "garage_spaces": garage_spaces,
    "neighborhood_quality": neighborhood_quality,
    "distance_to_downtown_mi": distance_to_downtown.round(1),
    "has_pool": has_pool,
    "overall_condition": overall_condition,
    "sale_price": sale_price.astype(int),
})

df.to_csv("data/housing_data.csv", index=False)
print(f"Generated {len(df)} rows -> data/housing_data.csv")
print(df.head())
