
import os
import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="✈️",
    layout="centered"
)


# ---------------------------------------------------------
# Application title
# ---------------------------------------------------------

st.title("✈️ Wellness Tourism Package Predictor")

st.write(
    """
    This application predicts whether a customer is likely to purchase
    the Wellness Tourism Package.

    Enter the customer information below and click **Predict Purchase**.
    """
)


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best_model.pkl"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# ---------------------------------------------------------
# Customer Details
# ---------------------------------------------------------

st.header("Customer Details")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

type_of_contact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

occupation = st.selectbox(
    "Occupation",
    [
        "Salaried",
        "Small Business",
        "Large Business",
        "Free Lancer"
    ]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

number_of_person_visiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=10,
    value=2
)

preferred_property_star = st.selectbox(
    "Preferred Property Star Rating",
    [3, 4, 5]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced", "Unmarried"]
)

number_of_trips = st.number_input(
    "Number of Trips per Year",
    min_value=0,
    max_value=30,
    value=2
)

passport = st.selectbox(
    "Has Passport?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

own_car = st.selectbox(
    "Owns a Car?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

number_of_children_visiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=10,
    value=0
)

designation = st.selectbox(
    "Designation",
    [
        "Executive",
        "Manager",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=0.0,
    value=30000.0
)


# ---------------------------------------------------------
# Customer Interaction Details
# ---------------------------------------------------------

st.header("Customer Interaction Details")

pitch_satisfaction_score = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

product_pitched = st.selectbox(
    "Product Pitched",
    [
        "Basic",
        "Standard",
        "Deluxe",
        "Super Deluxe",
        "King"
    ]
)

number_of_followups = st.number_input(
    "Number of Follow-ups",
    min_value=0,
    max_value=20,
    value=3
)

duration_of_pitch = st.number_input(
    "Duration of Pitch (minutes)",
    min_value=0.0,
    value=15.0
)


# ---------------------------------------------------------
# Convert user input into DataFrame
# ---------------------------------------------------------

input_data = pd.DataFrame(
    {
        "Age": [age],
        "TypeofContact": [type_of_contact],
        "CityTier": [city_tier],
        "DurationOfPitch": [duration_of_pitch],
        "Occupation": [occupation],
        "Gender": [gender],
        "NumberOfPersonVisiting": [number_of_person_visiting],
        "NumberOfFollowups": [number_of_followups],
        "ProductPitched": [product_pitched],
        "PreferredPropertyStar": [preferred_property_star],
        "MaritalStatus": [marital_status],
        "NumberOfTrips": [number_of_trips],
        "Passport": [passport],
        "PitchSatisfactionScore": [pitch_satisfaction_score],
        "OwnCar": [own_car],
        "NumberOfChildrenVisiting": [number_of_children_visiting],
        "Designation": [designation],
        "MonthlyIncome": [monthly_income]
    }
)


# ---------------------------------------------------------
# Display entered information
# ---------------------------------------------------------

with st.expander("View Entered Customer Data"):
    st.dataframe(input_data)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if st.button("Predict Purchase"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success(
            "The customer is likely to purchase the "
            "Wellness Tourism Package."
        )

    else:

        st.warning(
            "The customer is unlikely to purchase the "
            "Wellness Tourism Package."
        )

    st.write(
        f"Purchase Probability: **{probability:.2%}**"
    )
