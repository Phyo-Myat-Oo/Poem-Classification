import streamlit as st
import utility
import numpy as np

# Define label mapping
label_mapping = {
    1: "ကဗျာ",  # Poetry
    0: "စာသား"  # Prose
}

st.title("Myanmar NLP Text Classification with Deep Learning")
st.write("Enter text below to get a prediction.")

user_input = st.text_area("Enter text:", "")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        try:
            # Convert text if needed
            zawgyi_to_unicode = utility.detect_and_convert(user_input)
            
            # Transform text for model input
            input_vectorized = utility.transform_text(zawgyi_to_unicode)
            
            # Get prediction
            predictions = utility.model.predict(input_vectorized)
            
            predicted_class = int(predictions[0][0] > 0.5)
            
            
            # Get label text
            predicted_text = label_mapping.get(predicted_class, "Unknown")
            
            st.success(f"Predicted Class: **{predicted_text}**")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")