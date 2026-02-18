import streamlit as st
import joblib
import random
import datetime

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="National Cyber Crime Intelligence Portal",
    page_icon="🛡",
    layout="wide"
)

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

# ================= HEADER =================
col1, col2 = st.columns([1, 6])

with col1:
    st.image("assets/emblem.png", width=80)

with col2:
    st.markdown("### Government of India")
    st.markdown("#### National Cyber Crime Intelligence & Prediction Portal")

st.markdown("---")

# ================= NAVIGATION TABS =================
tab1, tab2, tab3 = st.tabs(["Register Complaint", "Track Complaint", "About System"])

# ==========================================================
# TAB 1 — REGISTER COMPLAINT
# ==========================================================
with tab1:

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

    if st.button("🔍 Predict Fraud Location"):

        with st.spinner("Analyzing complaint data using AI model..."):

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

            # Generate Complaint ID
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            complaint_id = f"NCCRP-{timestamp}-{random.randint(100,999)}"

        st.success(f"📍 Predicted High-Risk Location: {location[0]}")
        st.info(f"🆔 Complaint Registration ID: {complaint_id}")

# ==========================================================
# TAB 2 — TRACK COMPLAINT
# ==========================================================
with tab2:
    st.subheader("Track Your Complaint")

    complaint_input = st.text_input("Enter Complaint ID")

    if st.button("Track Status"):
        if complaint_input:
            st.success("Complaint Status: Under Investigation")
        else:
            st.warning("Please enter a valid Complaint ID")

# ==========================================================
# TAB 3 — ABOUT SYSTEM
# ==========================================================
with tab3:
    st.subheader("About the System")

    st.write("""
    This Cybercrime Intelligence System uses Machine Learning 
    to analyze complaint parameters and predict the most probable 
    fraud withdrawal hotspot.

    Model Used:
    - Random Forest Classifier

    Objective:
    - Assist cybercrime investigation authorities
    - Enable proactive intelligence generation
    - Improve response time in fraud cases
    """)

# ================= FOOTER =================
st.markdown("---")
st.caption("National Cyber Crime Intelligence Portal | Powered by AI & Streamlit")
