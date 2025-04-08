# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy everything to the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Streamlit's default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "diabetes_predictor_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
