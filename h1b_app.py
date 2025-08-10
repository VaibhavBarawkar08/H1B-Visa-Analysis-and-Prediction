import streamlit as st
import pandas as pd
import joblib
import base64

# Load trained model
model = joblib.load('xgboost_model2.pkl')

# Set page title
st.set_page_config(page_title="H1B Visa Approval Prediction", layout="centered")

# Function to load image and convert to base64
def get_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load your H1B visa image
img_base64 = get_base64("h1bvisa.jpg")

# --- Custom CSS ---
st.markdown(f"""
    <style>
        .stApp {{
            background-color: #FFE5B4; /* Peach */
        }}
        .banner {{
            background-image: url("data:image/jpeg;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            height: 200px;
            border-radius: 12px;
            margin-bottom: 20px;
        }}
        .main-card {{
            background-color: white;
            padding: 20px 30px;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
            max-width: 900px;
            margin: auto;
        }}
    </style>
""", unsafe_allow_html=True)

# --- Banner Image ---
st.markdown('<div class="banner"></div>', unsafe_allow_html=True)

# --- Card Start ---
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.title("H1B Visa Approval Prediction App")
st.markdown("Enter petition details to predict approval probability.")

# --- Two Columns Layout ---
col1, col2 = st.columns(2)

with col1:
    job_title = st.selectbox("Job Title Category", [
        "Engineering", "IT & Software", "Management", "Other",
        "Healthcare", "Research & Science", "Education", 
        "Finance & Accounting", "Marketing & Sales", "Legal"
    ])
    full_time = st.radio("Full-time Position?", ["Yes", "No"])
    new_employment = st.radio("New Employment?", ["Yes", "No"])
    continued_employment = st.radio("Continued Employment?", ["Yes", "No"])
    change_prev_employment = st.radio("Change Previous Employment?", ["Yes", "No"])

with col2:
    change_employer = st.radio("Change of Employer?", ["Yes", "No"])
    concurrent_employment = st.radio("Concurrent Employment?", ["Yes", "No"])
    amended_petition = st.radio("Amended Petition?", ["Yes", "No"])
    annual_wage = st.number_input("Annual Wage ($)", min_value=10000, max_value=300000, value=100000, step=1000)
    pw_wage_level = st.selectbox("PW Wage Level", ["I", "II", "III", "IV", "V"])
    employment_days = st.number_input("Employment Duration (days)", min_value=1, max_value=1460, value=1095)

# --- Predict Button ---
if st.button("Predict"):
    input_df = pd.DataFrame([{
        'JOB_TITLE': job_title,
        'FULL_TIME_POSITION': 1 if full_time == "Yes" else 0,
        'NEW_EMPLOYMENT': 1 if new_employment == "Yes" else 0,
        'CONTINUED_EMPLOYMENT': 1 if continued_employment == "Yes" else 0,
        'CHANGE_PREVIOUS_EMPLOYMENT': 1 if change_prev_employment == "Yes" else 0,
        'CHANGE_EMPLOYER': 1 if change_employer == "Yes" else 0,
        'NEW_CONCURRENT_EMPLOYMENT': 1 if concurrent_employment == "Yes" else 0,
        'AMENDED_PETITION': 1 if amended_petition == "Yes" else 0,
        'ANNUAL_WAGE': annual_wage,
        'PW_WAGE_LEVEL': pw_wage_level,
        'EMPLOYMENT_DURATION_DAYS': employment_days
    }])

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ Likely to be Approved (Confidence: {prob:.2%})")
    else:
        st.error(f"❌ Likely  to be Denied (Confidence: {1 - prob:.2%})")

# --- Card End ---
st.markdown('</div>', unsafe_allow_html=True)
