import streamlit as st
import joblib

model = joblib.load("cybercrime_model.pkl")

encoders = {}
columns = ["City","Crime_Type","Time_of_Crime","Victim_Age_Group",
           "Transaction_Mode","Bank_Type","Day_of_Week","Location"]

for col in columns:
    encoders[col] = joblib.load(f"{col}_encoder.pkl")

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
