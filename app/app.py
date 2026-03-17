import joblib
import numpy as np
import streamlit as st


model = joblib.load("app/app_model.pkl")

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 House Price Prediction App")
st.write("Enter house features below to predict the sale price.")

overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=300, max_value=6000, value=1500)
garage_cars = st.slider("Garage Capacity (cars)", 0, 5, 2)
total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=4000, value=800)
year_built = st.number_input("Year Built", min_value=1870, max_value=2025, value=2000)
full_bath = st.slider("Full Bathrooms", 0, 4, 2)
lot_area = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=100000, value=8000)

if st.button("Predict House Price"):
    features = np.array([[
        overall_qual,
        gr_liv_area,
        garage_cars,
        total_bsmt_sf,
        year_built,
        full_bath,
        lot_area
    ]])

    prediction_log = model.predict(features)[0]
    prediction_price = np.expm1(prediction_log)

    st.success(f"Predicted House Price: ${prediction_price:,.2f}")