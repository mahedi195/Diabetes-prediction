# Diabetes Prediction App ...

This is a machine learning application built to predict whether a person is likely to develop diabetes based on various health factors such as age, BMI, blood pressure, and more. deployed with a **Streamlit** interface.

## Features
- **Diabetes Prediction:** Based on user input (such as BMI, age, blood pressure, etc.), the app predicts if a person is likely to have diabetes or not.
- **Easy to Use Interface:** A simple, user-friendly web interface powered by Streamlit.
- **Dockerized Application:** The app is packaged into a Docker container, making it easy to deploy and run anywhere.

## Technologies Used
- **Streamlit**: A framework to quickly build interactive web applications for machine learning models.
- **Pandas**: For data handling and processing.
- **Scikit-learn**: For machine learning utilities and model evaluation.
- **Joblib**: For saving and loading models.
- **Docker**: For containerizing the application to make it portable and easy to deploy.

## Prerequisites

Before running the app, you need to have the following installed on your system:
- **Docker**: For running the app in a containerized environment.

## Setup Instructions

### Option 1: Running the App using docker

1. docker run -p 5000:5000 diabetes-app
   The app will be available in your browser at http://localhost:5000.



'''
# model training and save 
python train_and_save_models.py

# run from vs code
py -m streamlit run diabetes_predictor_app.py

# run from docker
docker info
docker build -t diabetes-app .

docker run -p 5000:5000 diabetes-app

# search it in web brower like google chrome or file explorer
http://localhost:5000

# run from cloud
share.streamlit.io  
'''
