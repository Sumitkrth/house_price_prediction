import streamlit as st
import pandas as pd
import joblib
import re

# Load model and data
model = joblib.load("models/house_price_model.pkl")
df = pd.read_csv("data/house_data.csv")

# ------------------------
# Helper function
# ------------------------
def clean_total_sqft(x):
    x = str(x).strip()
    if '-' in x:
        try:
            nums = list(map(float, x.split('-')))
            return (nums[0] + nums[1]) / 2
        except:
            return None
    cleaned = re.sub(r"[^\d.]", "", x)
    if cleaned.count('.') > 1:
        parts = cleaned.split('.')
        cleaned = parts[0] + '.' + ''.join(parts[1:])
    cleaned = cleaned.rstrip('.')
    try:
        return float(cleaned)
    except:
        return None

# Clean columns
df['total_sqft'] = df['total_sqft'].apply(clean_total_sqft)
df = df.dropna(subset=['total_sqft', 'price', 'location'])
df['location'] = df['location'].fillna("unknown").astype(str).str.strip().str.lower()

# Compute price_per_sqft safely
df['price_per_sqft'] = (df['price'].astype(float) * 100000) / df['total_sqft'].astype(float)

# Dropdown options
locations = sorted(df['location'].unique())

# Sidebar inputs
st.sidebar.header("House Details")
area = st.sidebar.number_input("Enter area in sqft", min_value=300.0, step=50.0)
bedrooms = st.sidebar.number_input("Number of Bedrooms", min_value=1, step=1)
location = st.sidebar.selectbox("Select Location", locations)

# Get avg price per sqft for location
avg_price_per_sqft = df.groupby('location')['price_per_sqft'].mean().to_dict()
overall_avg_pps = df['price_per_sqft'].mean()
price_per_sqft = avg_price_per_sqft.get(location, overall_avg_pps)

# Prediction
if st.sidebar.button("Predict Price"):
    input_df = pd.DataFrame([{
        'total_sqft': area,
        'Bedrooms': bedrooms,
        'location': location,
        'price_per_sqft': price_per_sqft
    }])
    predicted_price = model.predict(input_df)[0]
    st.success(f"💰 Predicted Price: {predicted_price:.2f} Lakh")
