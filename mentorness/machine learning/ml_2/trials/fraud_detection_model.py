import streamlit as st
import pandas as pd
import pickle

# Load the saved model
def load_model():
    with open('/internship/mentorness/machine learning/ml_2/SVM_Fast_best.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Make a prediction
def predict(model, input_data):
    return model.predict(input_data)

# Define the features
features = ['Transaction_Amount', 'Amount_paid', 'Vehicle_Type', 'TollBoothID',
            'Lane_Type', 'Vehicle_Dimensions', 'Geographical_Location',
            'Month', 'Week']

# Load the model
model = load_model()

# Create the Streamlit app
st.title('Fraud Detection Model')

# Create inputs for each feature
inputs = {}
for feature in features:
    inputs[feature] = st.text_input(f'Enter {feature}')

# When the 'Predict' button is clicked, make a prediction and display it
if st.button('Predict'):
    input_data = pd.DataFrame([list(inputs.values())], columns=features)
    prediction = predict(model, input_data)
    st.write(f'Prediction: {prediction[0]}')