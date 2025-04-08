# Diabetes Prediction App

This is a machine learning application built to predict whether a person is likely to develop diabetes based on various health factors such as age, BMI, blood pressure, and more. The model is trained using **XGBoost** and is deployed with a **Streamlit** interface.

## Features
- **Diabetes Prediction:** Based on user input (such as BMI, age, blood pressure, etc.), the app predicts if a person is likely to have diabetes or not.
- **Easy to Use Interface:** A simple, user-friendly web interface powered by Streamlit.
- **Dockerized Application:** The app is packaged into a Docker container, making it easy to deploy and run anywhere.

## Technologies Used
- **Streamlit**: A framework to quickly build interactive web applications for machine learning models.
- **XGBoost**: A powerful gradient boosting algorithm used for predictive modeling.
- **Pandas**: For data handling and processing.
- **Scikit-learn**: For machine learning utilities and model evaluation.
- **Joblib**: For saving and loading models.
- **Docker**: For containerizing the application to make it portable and easy to deploy.

## Prerequisites

Before running the app, you need to have the following installed on your system:
- **Docker**: For running the app in a containerized environment.

## Setup Instructions

### Option 1: Running the App using docker

1. docker run -p 8501:8501 diabetes-app
   The app will be available in your browser at http://localhost:8501.


