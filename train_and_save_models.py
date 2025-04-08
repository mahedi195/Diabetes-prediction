import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("diabetes.csv")

# Features and labels
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
log_model = LogisticRegression()
xgb_model = XGBClassifier(eval_metric='logloss')
svm_model = SVC(probability=True)
rf_model = RandomForestClassifier()

# Fit models
log_model.fit(X_train_scaled, y_train)
xgb_model.fit(X_train_scaled, y_train)
svm_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train_scaled, y_train)

# Accuracy (you can use these values in the app)
log_acc = accuracy_score(y_test, log_model.predict(X_test_scaled))
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test_scaled))
svm_acc = accuracy_score(y_test, svm_model.predict(X_test_scaled))
rf_acc = accuracy_score(y_test, rf_model.predict(X_test_scaled))

# Save models and scaler
joblib.dump(log_model, "log_model.pkl")
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print(f"✅ Models saved. Accuracies:\nLog: {log_acc:.2f}, XGB: {xgb_acc:.2f}, SVM: {svm_acc:.2f}, RF: {rf_acc:.2f}")




'''
python train_and_save_models.py
py -m streamlit run diabetes_predictor_app.py


docker run -p 8501:8501 diabetes-app
http://localhost:8501
'''
