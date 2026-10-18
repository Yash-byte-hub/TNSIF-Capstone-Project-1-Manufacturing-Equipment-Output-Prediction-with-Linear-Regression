# 🏭 Manufacturing Equipment Output Prediction

## 📌 Project Overview
This project is a Data Science Capstone focused on optimizing production efficiency for a manufacturing company operating injection molding machines. The objective is to develop a Supervised Learning (Regression) model to predict the hourly output (`Parts_Per_Hour`) based on machine operating parameters. By accurately forecasting production rates, the manufacturing team can optimize machine settings, improve scheduling, and proactively detect underperforming equipment.

## 📊 Dataset Information
* **Dataset:** `manufacturing_dataset_1000_samples.csv`
* **Records:** 1000 samples
* **Target Variable:** `Parts_Per_Hour` (Continuous numerical output)
* **Key Features:** Injection Temperature, Injection Pressure, Cycle Time, Material Viscosity, Ambient Temperature, Machine Age, Operator Experience, Shift, Machine Type, and Material Grade

## ⚙️ Methodology & Data Preprocessing
To prepare the data for the linear regression model, the following data wrangling pipeline was implemented:
* **Feature Engineering:** Extracted `Hour` from the `Timestamp` column before dropping the original string format.
* **Handling Missing Values:** Imputed missing values in `Material_Viscosity`, `Ambient_Temperature`, and `Operator_Experience` using the statistical median.
* **Categorical Encoding:** Applied One-Hot Encoding (`drop_first=True`) to `Shift`, `Machine_Type`, `Material_Grade`, and `Day_of_Week` to convert text attributes into numerical matrices without multicollinearity.
* **Feature Scaling:** Applied `StandardScaler` to numerical features to ensure uniform weight distribution during model training.

## 🤖 Model Performance
The Linear Regression model was trained and evaluated using an 80/20 Train-Test split.
* **R-squared Score:** `0.9056` (Successfully explains ~90.5% of the variance in production output)
* **Root Mean Squared Error (RMSE):** `3.509`
* **Mean Absolute Error (MAE):** `2.715`
* **Mean Squared Error (MSE):** `12.315`

## 🛠️ Technologies Used
* **Languages & Core Libraries:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* **Deployment & Containerization:** FastAPI, Uvicorn, Docker
* **Artifact Serialization:** Joblib (for `.pkl` model and scaler state preservation)

## 🚀 How to Run the Project Locally

### Option 1: Run the Jupyter Notebook
Open the `TNSIF Capstone Project 1.ipynb` file in Google Colab or a local Jupyter environment. Ensure the CSV dataset is loaded correctly to execute the exploratory data analysis (EDA), preprocessing, and model training cells.

### Option 2: Run the FastAPI Application via Docker
This repository includes a production-ready API endpoint using FastAPI and Docker.

**1. Clone the repository**
```bash
git clone https://github.com/YOUR-USERNAME/TNSIF-Capstone-Project-1-Manufacturing-Equipment-Output-Prediction-with-Linear-Regression.git
cd TNSIF-Capstone-Project-1-Manufacturing-Equipment-Output-Prediction-with-Linear-Regression
```

**2. Build the Docker Image**
```bash
docker build -t manufacturing-api .
```

**3. Run the Docker Container**
```bash
docker run -p 8000:8000 manufacturing-api
```

**4. Test the API**
Once the container is running, open your web browser and navigate to:
`http://localhost:8000/docs`

This will open the interactive Swagger UI where you can input machine parameters (like Temperature, Cycle Time, etc.) and receive real-time hourly output predictions.

## 📁 Repository Structure
* `TNSIF Capstone Project 1.ipynb`: The main notebook containing EDA, preprocessing, and model training code.
* `app.py`: The FastAPI application script for serving predictions.
* `Dockerfile`: Containerization instructions for deployment.
* `requirements.txt`: Python dependencies required for the Docker container.
* `linear_regression_model.pkl`: The serialized trained regression model.
* `scaler.pkl`: The serialized feature scaler fit on the training data.
* `feature_names.pkl`: The exact feature array structure required by the model.
