# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and model artifacts into the container
COPY app.py .
COPY linear_regression_model.pkl .
COPY scaler.pkl .
COPY feature_names.pkl .

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Command to run the FastAPI server using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
