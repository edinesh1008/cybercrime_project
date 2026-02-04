import streamlit as st
import joblib

# Load files
model = joblib.load("cybercrime_model.pkl")
city_encoder = joblib.load("city_encoder.pkl")
crime_encoder = joblib.load("crime_encoder.pkl")
location_encoder = joblib.load("location_encoder.pkl")

st.title("Cybercrime Prediction System")

# Show real names
city = st.selectbox("Select City", list(city_encoder.classes_))
crime = st.selectbox("Select Crime Type", list(crime_encoder.classes_))
amount = st.number_input("Enter Fraud Amount")

if st.button("Predict"):

    city_encoded = city_encoder.transform([city])[0]
    crime_encoded = crime_encoder.transform([crime])[0]

    prediction = model.predict([[city_encoded, crime_encoded, amount]])

    location_name = location_encoder.inverse_transform(prediction)[0]

    st.success(f"Predicted Crime Location: {location_name}")
