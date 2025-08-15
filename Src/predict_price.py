# src/predict_price.py

import joblib
import pandas as pd
import re
from utils import clean_total_sqft

# Load trained model and dataset
model = joblib.load("Models/house_price_model.pkl")
df = pd.read_csv("Data/house_data.csv")

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

# Figure out the correct area column name
if 'area' in df.columns:
    area_col = 'area'
elif 'total_sqft' in df.columns:
    area_col = 'total_sqft'
elif 'area (sqft)' in df.columns:
    df.rename(columns={'area (sqft)': 'area'}, inplace=True)
    area_col = 'area'
else:
    raise KeyError("No column found for area in dataset. Check CSV headers.")

# Function to clean and convert area to float
def clean_area(x):
    if isinstance(x, str):
        x = x.strip()
        # Handle ranges like "1200-1500"
        if '-' in x:
            parts = x.split('-')
            return (float(parts[0]) + float(parts[1])) / 2
        # Remove non-numeric characters like 'sqft'
        return float(re.sub(r"[^\d.]", "", x))
    return float(x)

# Clean area column
df[area_col] = df[area_col].apply(clean_total_sqft)
df = df.dropna(subset=[area_col])

# Compute price_per_sqft
df['price_per_sqft'] = df['price'] * 100000 / df[area_col]
avg_price_per_sqft = df.groupby('location')['price_per_sqft'].mean().to_dict()
overall_avg_pps = df['price_per_sqft'].mean()

# === User Input ===
area = float(input("Enter area in sqft: "))
bedrooms = int(input("Enter number of bedrooms: "))
location = input("Enter location: ").strip().lower()

# Get price_per_sqft from averages
price_per_sqft = avg_price_per_sqft.get(location, overall_avg_pps)

# Prepare input DataFrame
input_df = pd.DataFrame([{
    'total_sqft': area,
    'Bedrooms': bedrooms,
    'location': location,
    'price_per_sqft': price_per_sqft
}])

# Predict
predicted_price = model.predict(input_df)[0]

print(f"\n💰 Predicted House Price: {predicted_price:.2f} Lakh")
