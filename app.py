import streamlit as st
import joblib
import networkx as nx
import matplotlib.pyplot as plt

# ================= LOAD MODEL =================
model = joblib.load("cybercrime_model.pkl")

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

st.title("Cybercrime Investigation Dashboard")

# ================= USER DETAILS =================
name = st.text_input("Victim Name")
card_number = st.text_input("Card Number (12 digits only)")

# ================= DATE & TIME =================
crime_date = st.date_input("Select Crime Date")
crime_time = st.time_input("Select Crime Time")

month = crime_date.month
hour = crime_time.hour

# ================= CRIME INPUT =================
inputs = {}

for col in list(encoders.keys())[:-1]:
    inputs[col] = st.selectbox(co
