# src/train_model.py

import re
import pandas as pd
import numpy as np
from utils import clean_total_sqft
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

# ======================
# Helper Functions
# ======================

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

def remove_extreme_values(df):
    return df[
        (df.total_sqft >= 300) & 
        (df.total_sqft <= 10000) &
        (df.price >= 10) &
        (df.price <= 500)
    ]

# ======================
# Load & Preprocess Data
# ======================
df = pd.read_csv("Data/house_data.csv")

# Clean area
df['total_sqft'] = df['total_sqft'].apply(clean_total_sqft)

# Extract bedrooms
df['Bedrooms'] = df['size'].str.extract(r'(\d+)').astype(float)

# Drop missing values
df = df.dropna(subset=['total_sqft', 'Bedrooms', 'price', 'location'])

# Standardize location names
df['location'] = df['location'].str.strip().str.lower()

# Group rare locations
location_stats = df['location'].value_counts()
rare_locs = location_stats[location_stats <= 10].index
df['location'] = df['location'].apply(lambda x: 'other' if x in rare_locs else x)

# Add price_per_sqft
df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']  # assuming price is in lakhs

# Remove extreme outliers
df = remove_pps_outliers(df)
df = remove_extreme_values(df)

# ======================
# Features & Target
# ======================
X = df[['total_sqft', 'Bedrooms', 'location', 'price_per_sqft']]
y = df['price']

# Preprocessor
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['location']),
    ('num', 'passthrough', ['total_sqft', 'Bedrooms', 'price_per_sqft'])
])

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# ======================
# Train & Evaluate
# ======================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("✅ Model trained successfully!")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# ======================
# Save Model
# ======================
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/house_price_model.pkl")
print("📦 Model saved to models/house_price_model.pkl")
