"""
app.py — Streamlit Salary Predictor

Run locally:
    streamlit run app.py

Requires models/salary_model.pkl and models/model_columns.pkl
(generate with: python train_model.py)
"""

import os
import pandas as pd
import streamlit as st
import joblib

MODEL_PATH = "models/salary_model.pkl"
COLUMNS_PATH = "models/model_columns.pkl"
MODEL_NAME_PATH = "models/model_name.pkl"

st.set_page_config(page_title="Salary Predictor", layout="centered", page_icon="💼")
st.title("💼 Employee Salary Predictor")
st.caption("Final Year B.Tech Project — ML-based salary estimation")

if not (os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH)):
    st.error(
        "Model files not found. Please run `python train_model.py` first "
        "to generate the model in the models/ folder."
    )
    st.stop()

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)
model_name = joblib.load(MODEL_NAME_PATH) if os.path.exists(MODEL_NAME_PATH) else "ML Model"

st.sidebar.title("📁 Prediction Mode")
mode = st.sidebar.radio("Choose Mode", ["Single Prediction", "Batch Prediction"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Model in use: **{model_name}**")

# =======================
# SINGLE PREDICTION
# =======================
if mode == "Single Prediction":
    st.subheader("Enter Candidate Details")

    col1, col2 = st.columns(2)
    with col1:
        experience = st.number_input("Experience (years)", min_value=0, max_value=40, value=2)
        education_level = st.selectbox("Education Level", ["High School", "Bachelors", "Masters", "PhD"])
        job_role = st.selectbox(
            "Job Role",
            ["Software Engineer", "Data Scientist", "Analyst", "Manager", "Business Analyst", "DevOps Engineer"],
        )
    with col2:
        location = st.selectbox(
            "Location", ["Bangalore", "Delhi", "Mumbai", "Kolkata", "Chennai", "Hyderabad", "Pune"]
        )
        industry = st.selectbox(
            "Industry", ["IT", "Finance", "Healthcare", "Education", "Retail", "Manufacturing"]
        )

    input_df = pd.DataFrame([{
        "experience": experience,
        "education_level": education_level,
        "job_role": job_role,
        "location": location,
        "industry": industry,
    }])

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    if st.button("🔍 Predict Salary", use_container_width=True):
        prediction = model.predict(input_encoded)[0]
        st.success(f"💰 Predicted Annual Salary: ₹{round(float(prediction), 2):,.0f}")

# =======================
# BATCH PREDICTION
# =======================
else:
    st.subheader("📄 Upload CSV File for Batch Prediction")
    st.markdown("**CSV Format:** experience, education_level, job_role, location, industry")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            required_cols = {"experience", "education_level", "job_role", "location", "industry"}
            missing = required_cols - set(df.columns)
            if missing:
                st.error(f"❌ CSV is missing required columns: {missing}")
                st.stop()

            input_encoded = pd.get_dummies(df[list(required_cols)])
            input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

            predictions = model.predict(input_encoded)
            df["Predicted Salary"] = predictions.round(0)

            st.success("✅ Prediction Completed")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions",
                data=csv,
                file_name="predicted_salaries.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"❌ Error in processing file: {e}")
