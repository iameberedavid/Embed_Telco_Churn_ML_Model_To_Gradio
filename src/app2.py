import gradio as gr
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer

# Load the ML component to re-use in the app
def load_ml_components(fp):
    with open(fp, "rb") as f:
        loaded_ml_components = pickle.load(f)
    return loaded_ml_components

def predict_churn(gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines,
                  InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
                  StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                  tenure, MonthlyCharges, TotalCharges):
    
    # Load the ML components
    DIRPATH = os.path.dirname(os.path.realpath(__file__))
    ml_core_fp = os.path.join(DIRPATH, 'ml.pkl')
    loaded_ml_components = load_ml_components(fp=ml_core_fp)

    # Define the variables for each component
    encoder = loaded_ml_components['encoder']
    scaler = loaded_ml_components['scaler']
    model = loaded_ml_components['model']

    # Create a list of categorical features
    cat_features = [gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, InternetService,
                    OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
                    Contract, PaperlessBilling, PaymentMethod]
    
    # Create a list of categorical feature names
    cat_feature_names = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                         'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                         'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    # Ensure input values match encoder categories
    cat_features = [feature if feature in encoder.categories_[i] else encoder.categories_[i][0] for i, feature in enumerate(cat_features)]
    
    # Create a DataFrame with the categorical features
    cat_df = pd.DataFrame([cat_features], columns=cat_feature_names)
    
    # Fit the imputer with the categorical features
    imputer = SimpleImputer(strategy='most_frequent')
    imputer.fit(cat_df)
    
    # Impute missing values in categorical features
    imputed_cat_df = pd.DataFrame(imputer.transform(cat_df), columns=cat_feature_names)

    # Encode the imputed categorical features
    encoded_data = encoder.transform(imputed_cat_df)
    
    num_data = np.array([[tenure, MonthlyCharges, TotalCharges]])
    scaled_num_data = scaler.transform(num_data)
    
    # Combine encoded categorical features with scaled numerical features
    combined_data = np.hstack((encoded_data, scaled_num_data))
    
    # Load your training data (or synthetic data) in a format similar to the training data
    # Make sure this data is properly preprocessed like the training data
    
    # Fit the model with your training data
    # Replace X_train and y_train with your actual training data and labels
    model.fit(X_train, y_train)
    
    # Make prediction using the fitted model
    model_output = model.predict_proba(combined_data)
    prob_churn = float(model_output[0][1])
    prob_not_churn = 1 - prob_churn
    return {'Prediction Churn': prob_churn, 'Prediction Not Churn': prob_not_churn}

# Define the inputs
gender = gr.inputs.Radio(choices=['Male', 'Female'], label='Gender')
SeniorCitizen = gr.inputs.Radio(choices=['Yes', 'No'], label='SeniorCitizen')
Partner = gr.inputs.Radio(choices=['Yes', 'No'], label='Partner')
Dependents = gr.inputs.Radio(choices=['Yes', 'No'], label='Dependents')
PhoneService = gr.inputs.Radio(choices=['Yes', 'No'], label='PhoneService')
MultipleLines = gr.inputs.Radio(choices=['Yes', 'No'], label='MultipleLines')
InternetService = gr.inputs.Radio(choices=['Fiber optic', 'No', 'DSL'], label='InternetService')
OnlineSecurity = gr.inputs.Radio(choices=['Yes', 'No'], label='OnlineSecurity')
OnlineBackup = gr.inputs.Radio(choices=['Yes', 'No'], label='OnlineBackup')
DeviceProtection = gr.inputs.Radio(choices=['Yes', 'No'], label='DeviceProtection')
TechSupport = gr.inputs.Radio(choices=['Yes', 'No'], label='TechSupport')
StreamingTV = gr.inputs.Radio(choices=['Yes', 'No'], label='StreamingTV')
StreamingMovies = gr.inputs.Radio(choices=['Yes', 'No'], label='StreamingMovies')
Contract = gr.inputs.Radio(choices=['Month-to-month', 'One year', 'Two years'], label='Contract')
PaperlessBilling = gr.inputs.Radio(choices=['Yes', 'No'], label='PaperlessBilling')
PaymentMethod = gr.inputs.Radio(choices=['Electronic check', 'Mailed check', 'Credit card (automatic)', 'Bank transfer (automatic)'], label='PaymentMethod')
tenure = gr.inputs.Number(label='Tenure')
MonthlyCharges = gr.inputs.Number(label='MonthlyCharges')
TotalCharges = gr.inputs.Number(label='TotalCharges')

# Design the interface
gr.Interface(inputs=[gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines,
                     InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
                     StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                     tenure, MonthlyCharges, TotalCharges],
    outputs=gr.outputs.Label('Awaiting Submission...'),
    fn=predict_churn,
    title='Telco Churn Prediction',
    description='This app predicts whether a Telco customer will churn or not based on previous churn data.'
).launch(inbrowser=True, show_error=True, share=True)
