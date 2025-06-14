import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Title
st.title("Patient Condition Prediction from Drug Review 💊")

# Text input
user_review = st.text_area("Enter the patient's review:")

if st.button("Predict Condition"):
    if user_review.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Transform the input
        review_transformed = vectorizer.transform([user_review])
        
        # Predict
        prediction = model.predict(review_transformed)[0]
        
        # Mapping back the labels
        label_map = {0: "Depression", 1: "Diabetes, Type 2", 2: "High Blood Pressure"}
        
        st.success(f"Predicted Condition: **{label_map[prediction]}**")
