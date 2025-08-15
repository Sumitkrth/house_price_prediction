# streamlit_app.py
import streamlit as st
from Src.train_model import preprocess_and_train
from Src.predict_price import predict_house_price

st.title("🏠 House Price Predictor")

# Train section
if st.button("Train Model"):
    with st.spinner("Training model..."):
        preprocess_and_train()
    st.success("✅ Model trained successfully!")

# Prediction section
st.subheader("Make a Prediction")
area = st.number_input("Enter area (sqft):", min_value=300.0)
bedrooms = st.number_input("Enter number of bedrooms:", min_value=1, step=1)
location = st.text_input("Enter location:")

if st.button("Predict Price"):
    if area > 0 and bedrooms > 0 and location.strip():
        price = predict_house_price(area, bedrooms, location)
        st.success(f"💰 Predicted House Price: {price:.2f} Lakhs")
    else:
        st.error("Please fill in all fields.")
