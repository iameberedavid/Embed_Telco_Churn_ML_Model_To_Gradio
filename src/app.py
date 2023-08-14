# Load key libraries
import gradio as gr
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import AdaBoostClassifier

# Create key lists
expected_inputs = ['Gender', 'Senior_Citizen', 'Partner', 'Dependent', 'PhoneService', 'MultipleLines',
                   'InternetService', 'OnlineSecurity', 'OnlineBackup', 'StreamingMovies', 'Contract',
                   'PaperlessBilling', 'PaymentMethod', 'tenure', 'MonthlyCharges', 'TotalCharges']

# Useful functions
def load_ml_components(file_path = r"src\assets\ml"):
    """_summary_
    Args:
        file_path (reg, optional): _description_

    Returns:
    """
    with open(file_path, "rb") as file:
        loaded_ml_components = pickle.load(file)
    return loaded_ml_components

# Import the ml_component
loaded_ml_components = load_ml_components()

# Import the model
model = AdaBoostClassifier()
model.load_model()

# Function to process inputs and return prediction
# Inputs
Gender = 
Senior_Citizen =
Partner = 
Dependent = 
PhoneService = 
MultipleLines = 
InternetService = 
OnlineSecurity = 
OnlineBackup = 
StreamingMovies = 
Contract = 
PaperlessBilling = 
PaymentMethod = 
tenure = 
MonthlyCharges = 
TotalCharges = 


def interface_function(*args):
"""
"""

# SETUP
# Variables and Constants
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_fp = os.path.join(DIRPATH, "assets", "ml", "ml_components.pkl")

# Execution
ml_components_dict = load_ml_components(fp=ml_core_fp)
end2end_pipeline = ml_components_dict('pipeline')

print(f"\n[Info] ML components loaded: {list(ml_components_dict.keys())}")

# Interface
inputs = [gr.Dropdown(elem_id=i) for in range (17)] + [gr.Number(elem_id=i)
                                                       for i in range(4)]

demo = gr.Interface(
    interface_function,
    ['text'],
    "number",
    examples=[],
)

if __name__ == "__main__":
    demo.launch(debug=True)
