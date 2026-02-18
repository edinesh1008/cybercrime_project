import streamlit as st
import joblib

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="National Cyber Crime Intelligence Portal",
    page_icon="🛡",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.header {
    background-color:#003366;
    padding:15px;
    color:white;
    font-size:24px;
    font-weight:bold;
}
.subheader {
    background-color:#0059b3;
    padding:10px;
    color:white;
    font-size:18px;
}
.hero {
    background-color:#0b3d91;
    padding:40px;
    color:white;
    text-align:center;
    border-radius:10px;
}
.predict-box {
    padding:25px;
    background-color:#e6f0ff;
    border-radius:10px;
    font-size:22px;
    font-weight:bold;
    color:#00264d;
}
.footer {
    background-color:#003366;
    color:white;
    text-align:center;
    padding:10px;
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="header">Government of India | Ministry of Cyber Security</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">National Cyber Crime Intelligence & Prediction Portal</div>', unsafe_allow_html=True)

# ================= HERO SECTION =================
st.markdown("""
<div class="hero">
<h1>Cybercrime Predictive Intelligence System</h1>
<p>AI-Based Prediction of Likely Fraud Withdrawal Locations</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

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

# ================= INPUT SECTION =================
st.subheader("Register Cyber Fraud Case")

col1, col2, col3 = st.columns(3)

with col1:
    city = st.selectbox("City", encoders["City"].classes_)
    crime_type = st.selectbox("Crime Type", encoders["Crime_Type"].classes_)
    amount = st.number_input("Fraud Amount", min_value=1)

with col2:
    time_crime = st.selectbox("Time of Crime", encoders["Time_of_Crime"].classes_)
    age_group = st.selectbox("Victim Age Group", encoders["Victim_Age_Group"].classes_)
    transaction = st.selectbox("Transaction Mode", encoders["Transaction_Mode"].classes_)

with col3:
    bank = st.selectbox("Bank Type", encoders["Bank_Type"].classes_)
    day = st.selectbox("Day of Week", encoders["Day_of_Week"].classes_)
    month = st.number_input("Month", min_value=1, max_value=12)
    hour = st.number_input("Hour", min_value=0, max_value=23)

st.markdown("")

if st.button("🔍 Predict Fraud Location"):

    encoded_input = [
        encoders["City"].transform([city])[0],
        encoders["Crime_Type"].transform([crime_type])[0],
        amount,
        encoders["Time_of_Crime"].transform([time_crime])[0],
        encoders["Victim_Age_Group"].transform([age_group])[0],
        encoders["Transaction_Mode"].transform([transaction])[0],
        encoders["Bank_Type"].transform([bank])[0],
        encoders["Day_of_Week"].transform([day])[0],
        month,
        hour
    ]

    prediction = model.predict([encoded_input])
    location = encoders["Location"].inverse_transform(prediction)

    st.markdown("---")
    st.markdown(
        f'<div class="predict-box">📍 Predicted High-Risk Location: {location[0]}</div>',
        unsafe_allow_html=True
    )

# ================= FOOTER =================
st.markdown("---")
st.markdown('<div class="footer">Cybercrime Intelligence System | Powered by Machine Learning & Streamlit</div>', unsafe_allow_html=True)
