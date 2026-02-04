import streamlit as st
import joblib

# Load ML Model
model = joblib.load("cybercrime_model.pkl")

# Load Encoders (MATCHING YOUR FILE NAMES EXACTLY)
encoders = {
    "City": joblib.load("city_encoder.pkl"),
    "Crime_Type": joblib.load("Crime_Type_encoder.pkl"),
    "Time_of_Crime": joblib.load("Time_of_Crime_encoder.pkl"),
    "Victim_Age_Group": joblib.load("Victim_Age_Group_encoder.pkl"),
    "Transaction_Mode": joblib.load("Transaction_Mode_encoder.pkl"),
    "Bank_Type": joblib.load("Bank_Type_encoder.pkl"),
    "Day_of_Week": joblib.load("Day_of_Week_encoder.pkl"),
    "Location": joblib.load("location_encoder.pkl")
}

st.title("Cybercrime Prediction System")

# User Inputs
inputs = {}

for col in list(encoders.keys())[:-1]:
    inputs[col] = st.selectbox(f"Select {col}", encoders[col].classes_)

amount = st.number_input("Enter Fraud Amount", min_value=0)

if st.button("Predict"):

    encoded_input = []

    for col in list(encoders.keys())[:-1]:
        encoded_input.append(encoders[col].transform([inputs[col]])[0])

    # Insert Amount in correct position
    encoded_input.insert(2, amount)

    prediction = model.predict([encoded_input])

    result = encoders["Location"].inverse_transform(prediction)

    st.success(f"Predicted Crime Location: {result[0]}")
