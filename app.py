import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Import training libraries for self-healing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# --- Configuration ---
MODEL_FILE = 'random_forest_model.joblib'
DATA_FILE = 'CAR DETAILS FROM CAR DEKHO.csv'

# --- Function to Train Model (The Fix) ---
# def train_and_save_model():
#     """Retrains the model locally to ensure version compatibility."""
#     with st.spinner("Model version mismatch or missing detected. Retraining model locally..."):
#         try:
#             if not os.path.exists(DATA_FILE):
#                 st.error(f"Critical Error: '{DATA_FILE}' not found. Cannot retrain model.")
#                 st.stop()
                
#             df = pd.read_csv(DATA_FILE)
            
#             # Feature Engineering
#             current_year = 2025
#             df['Current_Age'] = current_year - df['year']
#             df = df.drop(['year', 'name'], axis=1)
            
#             X = df.drop('selling_price', axis=1)
#             y = df['selling_price']
            
#             # Preprocessing
#             # Note: We use the EXACT column names from the CSV
#             categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']
            
#             preprocessor = ColumnTransformer(
#                 transformers=[
#                     ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#                 ],
#                 remainder='passthrough'
#             )
            
#             model = Pipeline(steps=[
#                 ('preprocessor', preprocessor),
#                 ('model', RandomForestRegressor(n_estimators=100, random_state=42))
#             ])
            
#             model.fit(X, y)
#             joblib.dump(model, MODEL_FILE)
#             st.success("Model retrained and saved successfully! Loading app...")
#             return model
            
#         except Exception as e:
#             st.error(f"Failed to retrain model: {e}")
#             st.stop()

# --- 1. Load the Saved Model ---
try:
    model = joblib.load(MODEL_FILE)
except (FileNotFoundError, ValueError, Exception) as e:
    model = train_and_save_model()

# --- 2. App Layout ---
st.title('🚗 Used Car Price Predictor')
st.markdown("Enter the car details below to get an estimated selling price.")

st.header("Car Details")

col1, col2 = st.columns(2)

with col1:
    # Note: Removed 'Present_Price' because it is not in the training dataset
    
    kms_driven = st.number_input('Kilometers Driven', min_value=0, max_value=500000, value=30000, step=1000)
    
    fuel_type = st.selectbox('Fuel Type', ('Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'))
    
    seller_type = st.selectbox('Seller Type', ('Individual', 'Dealer', 'Trustmark Dealer'))

with col2:
    transmission_type = st.selectbox('Transmission Type', ('Manual', 'Automatic'))
    
    # IMPORTANT: Used exact strings "First Owner" etc. to match training data
    owner = st.selectbox('Number of Previous Owners', 
                         ('First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'))

    year = st.number_input('Year of Purchase', min_value=1990, max_value=2025, value=2015, step=1)
    current_age = 2025 - year 

# --- 3. Prediction Logic ---
if st.button('Predict Price', type="primary"):
    # Create dictionary with EXACT column names as in the CSV/Training
    input_data = {
        'km_driven': kms_driven,          # Matches 'km_driven' in CSV
        'fuel': fuel_type,                # Matches 'fuel' in CSV
        'seller_type': seller_type,       # Matches 'seller_type' in CSV
        'transmission': transmission_type, # Matches 'transmission' in CSV
        'owner': owner,                   # Matches 'owner' in CSV
        'Current_Age': current_age        # Matches our engineered feature
    }

    input_df = pd.DataFrame([input_data])
    
    st.subheader("Input Parameters")
    st.dataframe(input_df)

    try:
        prediction = model.predict(input_df)
        predicted_price = prediction[0]
        
        st.success(f'Predicted Selling Price: ₹ {predicted_price:,.2f} Lakhs')
        st.info(f"(Approximately ₹ {predicted_price * 100000:,.0f})")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")