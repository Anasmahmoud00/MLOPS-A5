# check_threshold.py
import mlflow
import sys

mlflow.set_tracking_uri("file:///C:/Users/Rihana Nasr/Desktop/MLOPS-A5/mlruns")

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

run = mlflow.get_run(run_id)
accuracy = run.data.metrics.get("accuracy")

if accuracy is None:
    print(f"❌ Accuracy not found for Run ID {run_id}")
    sys.exit(1)

print(f"ℹ️ Accuracy for Run ID {run_id}: {accuracy:.4f}")

threshold = 0.85
if accuracy < threshold:
    print(f"❌ Accuracy below threshold ({threshold}) — deployment stopped!")
    sys.exit(1)
else:
    print(f"✅ Accuracy above threshold ({threshold}) — ready to deploy!")