import os
import mlflow
import sys

# Get the Run ID
with open("model_info.txt") as f:
    run_id = f.read().strip()

# Set your MLflow Tracking URI
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)

# Get the run data
run = mlflow.get_run(run_id)
accuracy = run.data.metrics.get("accuracy", 0)

print(f"Run ID: {run_id}, Accuracy: {accuracy}")

# Threshold check
if accuracy < 0.85:
    print("Accuracy below 0.85 – failing deployment")
    sys.exit(1)

print("Accuracy threshold met – proceeding to deploy")