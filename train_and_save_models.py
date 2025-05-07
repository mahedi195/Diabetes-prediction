import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
svm_model = SVC(probability=True)
rf_model = RandomForestClassifier()

log_model.fit(X_train_scaled, y_train)
svm_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train_scaled, y_train)

log_acc = accuracy_score(y_test, log_model.predict(X_test_scaled))
svm_acc = accuracy_score(y_test, svm_model.predict(X_test_scaled))
rf_acc = accuracy_score(y_test, rf_model.predict(X_test_scaled))

joblib.dump(log_model, "log_model.pkl")
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Models saved. Accuracies:")
print(f"Logistic Regression: {log_acc:.2f}")
print(f"SVM: {svm_acc:.2f}")
print(f"Random Forest: {rf_acc:.2f}")
