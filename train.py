# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("Iris_Classification")

data = pd.read_csv("iris.csv")
X = data.drop("species", axis=1)
y = data["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

with mlflow.start_run() as run:
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    
    joblib.dump(model, "model.pkl")
    print("✅ Model saved as model.pkl and logged to MLflow run:", run.info.run_id)
    
    # Save Run ID for GitHub Actions
    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)