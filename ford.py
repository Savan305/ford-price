import streamlit as st
import pandas as pd
import joblib

# Load model
data = joblib.load("ford_price_model.pkl")
model = data["model"]
scaler = data["scaler"]
label_encoders = data["label_encoders"]
columns = data["columns"]

st.set_page_config(page_title="Ford Car Price Prediction", layout="centered")

st.title("🚗 Ford Car Price Prediction App")

st.write("Enter car details to predict the price")

# =============================
# User Inputs
# =============================
model_name = st.selectbox("Car Model", label_encoders["model"].classes_)
year = st.number_input("Year", min_value=2000, max_value=2025, value=2018)
transmission = st.selectbox("Transmission", label_encoders["transmission"].classes_)
mileage = st.number_input("Mileage (km)", min_value=0, value=30000)
fuel_type = st.selectbox("Fuel Type", label_encoders["fuelType"].classes_)
tax = st.number_input("Tax", min_value=0, value=150)
mpg = st.number_input("MPG", value=50.0)
engine_size = st.number_input("Engine Size (L)", value=1.5)

# =============================
# Prediction
# =============================
if st.button("Predict Price 💰"):
    input_data = pd.DataFrame([{
        "model": model_name,
        "year": year,
        "transmission": transmission,
        "mileage": mileage,
        "fuelType": fuel_type,
        "tax": tax,
        "mpg": mpg,
        "engineSize": engine_size
    }])

    # Label Encoding
    for col, le in label_encoders.items():
        input_data[col] = le.transform(input_data[col])

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    st.success(f"💰 Estimated Car Price: ₹ {int(prediction[0]):,}")

