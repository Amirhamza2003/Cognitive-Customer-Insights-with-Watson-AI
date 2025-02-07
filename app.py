import streamlit as st
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError

# Define custom objects (specifically for loss functions or metrics)
def custom_objects():
    return {'mse': MeanSquaredError()}

# Load the trained model with custom loss function
model = tf.keras.models.load_model("D:\\ibm\\autoencoder_model (2).h5", custom_objects={'mse': MeanSquaredError})

# Function to handle prediction
def predict_segment(age, income, spending_score):
    input_data = np.array([[age, income, spending_score]])
    encoded_data = model.predict(input_data)  # Get encoded features
    return encoded_data

# Set up Streamlit interface
st.title('Customer Segmentation Using Deep Clustering')

# Input fields for user data
age = st.number_input('Enter Age', min_value=18, max_value=100, value=30)
income = st.number_input('Enter Income', min_value=1000, max_value=100000, value=50000)
spending_score = st.number_input('Enter Spending Score', min_value=0, max_value=100, value=50)

# Button for prediction
if st.button('Get Customer Segmentation'):
    encoded_data = predict_segment(age, income, spending_score)
    st.write(f"Encoded Data: {encoded_data}")
