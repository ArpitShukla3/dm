import streamlit as st
import pandas as pd
import joblib
import numpy as np


try:
    model = joblib.load('random_forest_model.joblib')
    st.write("Model loaded successfully!")
except Exception as e:
    st.write(f"Error loading model: {e}")
    st.stop()

st.title('🚗 Used Car Price Predictor')
st.write("Enter the car details below to get a price prediction.")



st.header("Car Details")


col1, col2 = st.columns(2)

with col1:

    present_price = st.number_input('Current Showroom Price (in ₹ Lakhs)', min_value=0.1, max_value=50.0, value=5.0, step=0.1)
    

    kms_driven = st.number_input('Kilometers Driven', min_value=0, max_value=500000, value=30000, step=1000)
    

    fuel_type = st.selectbox('Fuel Type', ('Petrol', 'Diesel', 'CNG'))
    
    seller_type = st.selectbox('Seller Type', ('Individual', 'Dealer'))

with col2:
    transmission_type = st.selectbox('Transmission Type', ('Manual', 'Automatic'))
    owner = st.selectbox('Number of Previous Owners', (0, 1, 2, 3)) # Based on common data values
    year = st.number_input('Year of Purchase', min_value=1990, max_value=2022, value=2015, step=1)
    current_age = 2022 - year

if st.button('Predict Price'):
    input_data = {
        'Present_Price': present_price,
        'Kms_Driven': kms_driven,
        'Fuel_Type': fuel_type,
        'Seller_Type': seller_type,
        'Transmission': transmission_type,
        'Owner': owner,
        'Current_Age': current_age
    }

    input_df = pd.DataFrame([input_data])
    
    st.write("--- Input Data ---")
    st.dataframe(input_df)

    try:
        prediction = model.predict(input_df)
        

        st.success(f'Predicted Selling Price: ₹ {prediction[0]:,.2f} Lakhs')
        
        st.info(f"(Approximately ₹ {prediction[0] * 100000:,.0f})")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")