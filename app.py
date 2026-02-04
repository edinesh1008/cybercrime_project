import streamlit as st
import joblib

# Load model
model = joblib.load("cybercrime_model.pkl")

# Load encoders (Lowercase safe version)
encoders = {
    "City": joblib.load("city_encoder.pkl"),
    "Crime_Type": joblib.load("crime_type_encoder.pkl"),
    "Time_of_Crime": joblib.load("time_of_crime_encoder.pkl"),
    "Victim_Age_Group": joblib.load("victim_age_group_encoder.pkl"),
    "Transaction_Mode": joblib.load("transaction_mode_encoder.pkl"),
    "Bank_Type": joblib.load("bank_type_encoder.pkl"),
    "Day_of_Week": joblib.load("day_of_week_encoder.pkl"),
    "Location": joblib.load("location_encoder.pkl")
}

st.title("Cybercrime Prediction System")

# User Inputs
inputs = {}

for col in list(encoders.keys())[:-1]:
    inputs[col] = st.selectbox(col, encoders[col].classes_)

amount = st.number_input("Enter Fraud Amount")

if st.button("Predict"):

    encoded_input = []

    for col in list(encoders.keys())[:-1]:
        encoded_input.append(encoders[col].transform([inputs[col]])[0])

    encoded_input.insert(2, amount)

    prediction = model.predict([encoded_input])

    result = encoders["Location"].inverse_transform(prediction)

    st.success(f"Predicted Location: {result[0]}")
