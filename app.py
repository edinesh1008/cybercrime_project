import streamlit as st
import joblib

model = joblib.load("cybercrime_model.pkl")

encoders = {}
columns = ["City","Crime_Type","Time_of_Crime","Victim_Age_Group",
           "Transaction_Mode","Bank_Type","Day_of_Week","Location"]

for col in columns:
    encoders = {
    "City": joblib.load("City_encoder.pkl"),
    "Crime_Type": joblib.load("Crime_Type_encoder.pkl"),
    "Time_of_Crime": joblib.load("Time_of_Crime_encoder.pkl"),
    "Victim_Age_Group": joblib.load("Victim_Age_Group_encoder.pkl"),
    "Transaction_Mode": joblib.load("Transaction_Mode_encoder.pkl"),
    "Bank_Type": joblib.load("Bank_Type_encoder.pkl"),
    "Day_of_Week": joblib.load("Day_of_Week_encoder.pkl"),
    "Location": joblib.load("Location_encoder.pkl")
}


st.title("Cybercrime Prediction System")

inputs = {}

for col in columns[:-1]:
    inputs[col] = st.selectbox(col, encoders[col].classes_)

amount = st.number_input("Amount")

if st.button("Predict"):

    encoded_input = []

    for col in columns[:-1]:
        encoded_input.append(encoders[col].transform([inputs[col]])[0])

    encoded_input.insert(2,amount)

    prediction = model.predict([encoded_input])

    result = encoders["Location"].inverse_transform(prediction)

    st.success(f"Predicted Location: {result[0]}")
