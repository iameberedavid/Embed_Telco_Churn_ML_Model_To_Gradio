# Load key libraries
import gradio as gr
import pandas as pd
import numpy as np
import pickle
import os

def predict_churn(gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines,
                  InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
                  StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                  tenure, MonthlyCharges, TotalCharges):
    
    # Load the ML components
    DIRPATH = os.path.dirname(os.path.realpath(__file__))
    ml_core_fp = os.path.join(DIRPATH, 'ml_components.pkl')
    with open(ml_core_fp, "rb") as f:
        loaded_model_components = pickle.load(f)
    
    # Extract the ML components
    model = loaded_model_components['model']
    encoder = loaded_model_components['encoder']
    scaler = loaded_model_components['scaler']

    # Create a list of categorical features
    cat_features = [gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, InternetService,
                    OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
                    Contract, PaperlessBilling, PaymentMethod]

    # Encode the categorical features
    encoded_data = encoder.transform([cat_features])
    
    num_data = np.array([[tenure, MonthlyCharges, TotalCharges]])
    scaled_num_data = scaler.transform(num_data)
    
    combined_data = np.hstack((encoded_data, scaled_num_data))
    
    # Make prediction using the fitted model
    model_output = model.predict_proba(combined_data)
    prob_churn = float(model_output[0][1])
    return {'Churn Probability': prob_churn}

# Define the inputs
gender = gr.Radio(choices=['Male', 'Female'], label='Gender')
SeniorCitizen = gr.Radio(choices=['Yes', 'No'], label='SeniorCitizen')
Partner = gr.Radio(choices=['Yes', 'No'], label='Partner')
Dependents = gr.Radio(choices=['Yes', 'No'], label='Dependents')
PhoneService = gr.Radio(choices=['Yes', 'No'], label='PhoneService')
MultipleLines = gr.Radio(choices=['Yes', 'No'], label='MultipleLines')
InternetService = gr.Radio(choices=['Fiber optic', 'No', 'DSL'], label='InternetService')
OnlineSecurity = gr.Radio(choices=['Yes', 'No'], label='OnlineSecurity')
OnlineBackup = gr.Radio(choices=['Yes', 'No'], label='OnlineBackup')
DeviceProtection = gr.Radio(choices=['Yes', 'No'], label='DeviceProtection')
TechSupport = gr.Radio(choices=['Yes', 'No'], label='TechSupport')
StreamingTV = gr.Radio(choices=['Yes', 'No'], label='StreamingTV')
StreamingMovies = gr.Radio(choices=['Yes', 'No'], label='StreamingMovies')
Contract = gr.Radio(choices=['Month-to-month', 'One year', 'Two year'], label='Contract')
PaperlessBilling = gr.Radio(choices=['Yes', 'No'], label='PaperlessBilling')
PaymentMethod = gr.Radio(choices=['Electronic check', 'Mailed check', 'Credit card (automatic)', 'Bank transfer (automatic)'], label='PaymentMethod')
tenure = gr.Number(label='Tenure')
MonthlyCharges = gr.Number(label='MonthlyCharges')
TotalCharges = gr.Number(label='TotalCharges')

# Design the interface
gr.Interface(inputs=[gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines,
                     InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
                     StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                     tenure, MonthlyCharges, TotalCharges],
    outputs=gr.Label('Awaiting Submission...'),
    fn=predict_churn,
    title='Telco Churn Prediction',
    description='This app predicts whether a Telco customer will churn or not based on previous churn data.'
).launch(inbrowser=True, show_error=True, share=True)
