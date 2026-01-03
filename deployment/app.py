import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Wellness Tourism Purchase Predictor", layout="centered")

st.title("Wellness Tourism Package Purchase Prediction")
st.write(
    "Enter customer and interaction details to predict the probability that the customer will purchase the new Wellness Tourism package."
)

# --- Load model from HF Model Hub ---
MODEL_REPO = "NehaRai22/Wellness_Tourism_Prediction/tourism_package_prediction_model" # Corrected to the model repo ID where the model was uploaded
MODEL_FILE = "best_tourism_ProdTaken_v1.joblib"  # ensure train.py uploads this exact name

@st.cache_resource
def load_model():
    local_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, repo_type="model")
    return joblib.load(local_path)

model = load_model()

# --- UI: collect inputs aligned to columns in  CSV ---
# Categorical picklists based on observed values in the dataset
typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
citytier = st.selectbox("City Tier", ["1", "2", "3"])  # ordinal treated as category
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])  # observed in data
preferred_star = st.selectbox("Preferred Property Star", ["1", "2", "3", "4", "5"])  # ordinal category
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

# Numeric features / counts
age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
num_persons = st.number_input("Number Of Person Visiting", min_value=1, max_value=10, value=2, step=1)
num_trips = st.number_input("Number Of Trips (avg per year)", min_value=0, max_value=50, value=2, step=1)
num_children = st.number_input("Number Of Children Visiting (<=5 years)", min_value=0, max_value=10, value=0, step=1)
monthly_income = st.number_input("Monthly Income (₹)", min_value=1000, max_value=100000, value=18000, step=500)

# Binary flags
passport = st.selectbox("Passport available?", ["No", "Yes"])
owncar = st.selectbox("Own car?", ["No", "Yes"])

# Interaction data
pitch_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
num_followups = st.number_input("Number Of Followups", min_value=0, max_value=20, value=3, step=1)
duration_pitch = st.number_input("Duration Of Pitch (minutes)", min_value=0, max_value=120, value=10, step=1)

# --- Assemble input into a DataFrame with exact column names ---
def to_bool(x):
    return 1 if x == "Yes" else 0

input_row = {
    # Customer Details
    "Age": age,
    "TypeofContact": typeofcontact,
    "CityTier": citytier,                       # keep as string if model pipeline does OneHot
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "PreferredPropertyStar": preferred_star,    # keep as string if OneHot
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": to_bool(passport),
    "OwnCar": to_bool(owncar),
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,

    # Customer Interaction Data
    "PitchSatisfactionScore": int(pitch_score),
    "ProductPitched": product_pitched,
    "NumberOfFollowups": num_followups,
    "DurationOfPitch": duration_pitch,
}

input_df = pd.DataFrame([input_row])

st.subheader("Input Preview")
st.dataframe(input_df)

if st.button("Predict Purchase Probability"):
    try:
        # If the saved model is a Pipeline, this will handle preprocessing internally.
        proba = model.predict_proba(input_df)[0, 1]
        pred = int(proba >= 0.5)

        st.success(f"Predicted probability of purchase: **{proba:.2%}**")
        st.write(f"Prediction: **{'Will Purchase (1)' if pred == 1 else 'Will Not Purchase (0)'}**")
    except Exception as e:
        st.error("Prediction failed. Ensure your trained model is a pipeline that accepts raw features.")
        st.exception(e)
        st.info("If your model expects label-encoded features, either re-train with a preprocessing pipeline "
                "or add identical encoders in the app before prediction.")
