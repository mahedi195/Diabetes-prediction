FROM python:3.13-slim

WORKDIR /app

# Copy and install dependencies first for caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app and model files
COPY diabetes_predictor_app.py /app/
COPY log_model.pkl svm_model.pkl rf_model.pkl scaler.pkl /app/

EXPOSE 5000

CMD ["streamlit", "run", "diabetes_predictor_app.py", "--server.port=5000", "--server.address=0.0.0.0"]
