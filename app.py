from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# 1. Load the saved model and preprocessing artifacts
model = joblib.load('linear_regression_model.pkl')
scaler = joblib.load('scaler.pkl')
expected_features = joblib.load('feature_names.pkl')

# 2. Initialize FastAPI app
app = FastAPI(title="Manufacturing Output Prediction API")

# 3. Define the expected input data structure (matching your original dataset features)
class MachineData(BaseModel):
    Injection_Temperature: float
    Injection_Pressure: float
    Cycle_Time: float
    Cooling_Time: float
    Material_Viscosity: float
    Ambient_Temperature: float
    Machine_Age: float
    Operator_Experience: float
    Maintenance_Hours: float
    Shift: str
    Machine_Type: str
    Material_Grade: str
    Day_of_Week: str
    Temperature_Pressure_Ratio: float
    Total_Cycle_Time: float
    Efficiency_Score: float
    Machine_Utilization: float
    Hour: int

@app.post("/predict")
def predict_output(data: MachineData):
    # Convert incoming JSON data into a pandas DataFrame
    df_input = pd.DataFrame([data.dict()])
    
    # Apply One-Hot Encoding
    categorical_cols = ['Shift', 'Machine_Type', 'Material_Grade', 'Day_of_Week']
    df_encoded = pd.get_dummies(df_input, columns=categorical_cols)
    
    # Align the new data with the training features (adds missing dummy columns as 0)
    for col in expected_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Ensure exact column order
    df_encoded = df_encoded[expected_features]
    
    # Scale the features
    scaled_features = scaler.transform(df_encoded)
    
    # Make prediction
    prediction = model.predict(scaled_features)
    
    # Return the predicted output, rounded to 2 decimal places
    return {"Predicted_Parts_Per_Hour": round(float(prediction[0]), 2)}
