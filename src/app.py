# Load key libraries
import gradio as gr
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import AdaBoostClassifier

# Create key lists
expected_inputs = ['gender', 'SeniorCitizen', 'Partner', 'Dependent', 'PhoneService', 'MultipleLines', 
                   'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                   'PaperlessBilling', 'PaymentMethod', 'tenure', 'MonthlyCharges', 'TotalCharges']

# Function to load Machine Learning components
def load_ml_components(file_path):
    # Load the ML component to re-use in the app
    with open(file_path, 'rb') as file:
        loaded_ml_components = pickle.load(file)
    return loaded_ml_components

# Load the ML components
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_file_path = os.path.join(DIRPATH, 'assets', 'ml', 'ml_components.pkl')
loaded_ml_components = load_ml_components(file_path = ml_core_file_path)

# Define the variable for each component
encoder = loaded_ml_components['encoder']
scaler = loaded_ml_components['scaler']
model = loaded_ml_components['best model']

def predict_churn(*args, encoder=encoder, scaler=scaler, model=model):

    input_data = pd.DataFrame([args], columns = expected_inputs)

    # Encode the data
    num_col = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_col = ['gender', 'SeniorCitizen', 'Partner', 'Dependent', 'PhoneService', 'MultipleLines',
               'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
               'TechSupport', 'StreamingTV','StreamingMovies', 'Contract',
               'PaperlessBilling', 'PaymentMethod']
    
    cat_col = cat_col.astype(str)
    encoded_data = encoder.transform(cat_col)
    encoded_df = pd.concat([num_col, encoded_data], axis=1)

    # Impute missing values
    # imputed_df = imputer.transform(encoded_df)

    # Scale the data
    scaled_df = scaler.transform(encoded_df)

    # Prediction
    model_output = model.predict_proba(scaled_df)

    # Probability of Churn (positive class)
    prob_churn = float(model_output[0][1])

    # Probability of Not churn (negative class)
    prob_not_churn = 1 - prob_churn
    return{'Prediction Churn': prob_churn,
           'Prediction Not Churn': prob_not_churn}

# Define the inputs
gender = gr.Radio(choices=['Male', 'Female'], label='Gender')
SeniorCitizen = gr.Radio(choices=['Yes', 'No'], label='SeniorCitizen')
Partner = gr.Radio(choices=['Yes', 'No'], label='Partner')
Dependent = gr.Radio(choices=['Yes', 'No'], label='Dependent')
PhoneService = gr.Radio(choices=['Yes', 'No'], label='PhoneService')
MultipleLines = gr.Radio(choices=['Yes', 'No'], label='MultipleLines')
InternetService = gr.Radio(choices=['Fiber optic', 'No', 'DSL'], label='InternetService')
OnlineSecurity = gr.Radio(choices=['Yes', 'No'], label='OnlineSecurity')
OnlineBackup = gr.Radio(choices=['Yes', 'No'], label='OnlineBackup')
DeviceProtection = gr.Radio(choices=['Yes', 'No'], label='viceProtection')
TechSupport = gr.Radio(choices=['Yes', 'No'], label='TechSupport')
StreamingTV = gr.Radio(choices=['Yes', 'No'], label='StreamingTV')
StreamingMovies = gr.Radio(choices=['Yes', 'No'], label='StreamingMovies')
Contract = gr.Radio(choices=['Month-to-month', 'One year', 'Two years'], label='Contract')
PaperlessBilling = gr.Radio(choices=['Yes', 'No'], label='PaperlessBilling')
PaymentMethod = gr.Radio(choices=['Electronic check', 'Mailed check', 'Credit card (automatic)', 'Bank transfer (automatic)'], label='PaymentMethod')
tenure = gr.Number(label='Tenure')
MonthlyCharges = gr.Number(label='MonthlyCharges')
TotalCharges = gr.Number(label='TotalCharges')

# Design the interface
gr.Interface(inputs=expected_inputs,
    outputs=gr.Label('Awaiting Submission...'),
    fn=predict_churn,
    title='Telco Churn Prediction',
    description='This app predicts whether a Telco customer will churn or not'
).launch(inbrowser=True, show_error=True, share=True)