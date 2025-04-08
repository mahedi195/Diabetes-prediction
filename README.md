# Diabetes Prediction 

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

Before running the app, we need to have the following installed on your system:
- **Docker**: For running the app in a containerized environment.

## Setup Instructions

### Option 1: Running the App using docker

1. docker run -p 8501:8501 diabetes-app
   The app will be available browser at http://localhost:8501.




   '''
push the code to github
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/mahedi195/diabetes-predictor-app.git
git push -u origin main


if we want to chane somewhere then upload project to github
git  status                                                                              #checking status
git add .                                                                                #Stage the changes
git commit -m "Dockerized app, added Dockerfile and updated the app files"               #commit changes
git push origin main                                                                     #push the changes to github

'''



