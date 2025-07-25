




import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the saved model and encoders
model = joblib.load("svm_brand_model.pkl")  # SVM model
le_sub = joblib.load("subcategory_encoder.pkl")  # Subcategory encoder
label_encoder = joblib.load("brand_label_encoder.pkl")  # Brand encoder

# List of subcategories for dropdown
subcategories = list(le_sub.classes_)

# UI
st.set_page_config(page_title="LuxeLens Handbag Brand Predictor", layout="centered")
st.title("👜 LuxeLens: Predict Your Handbag Brand")

# Form
with st.form("prediction_form"):
    st.markdown("### Enter Handbag Details")

    subcategory = st.selectbox("Select Subcategory", subcategories)
    price = st.slider("Price (₹)", 500, 3000, 1500)
    rating = st.slider("Rating", 0.0, 5.0, 3.5, 0.1)

    submit = st.form_submit_button("Predict Brand")

if submit:
    # Encode inputs
    sub_encoded = le_sub.transform([subcategory])[0]
    user_input = np.array([[sub_encoded, price, rating]])

    # Make prediction
    probs = model.predict_proba(user_input)[0]
    predicted_index = np.argmax(probs)
    predicted_brand = label_encoder.inverse_transform([predicted_index])[0]

    # Format probabilities
    brand_probs = {
        label_encoder.inverse_transform([i])[0]: round(p * 100, 2)
        for i, p in enumerate(probs)
    }

    # Output
    st.success(f"🧠 Predicted Brand: **{predicted_brand}**")

    st.markdown("#### 🔍 Prediction Probabilities")
    st.dataframe(pd.DataFrame({
        "Brand": list(brand_probs.keys()),
        "Probability (%)": list(brand_probs.values())
    }))

