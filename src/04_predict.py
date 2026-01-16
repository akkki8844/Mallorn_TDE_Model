import numpy as np
import pandas as pd
import joblib
import os

print("Running 04_predict.py")

test = pd.read_csv("data/processed/test_features.csv")
X_test = test.drop(["object_id"], axis=1)

models = []
for i in range(5):
    models.append(joblib.load(f"models/lgbm_fold_{i}.pkl"))

threshold = np.load("models/best_threshold.npy").item()
print("Using threshold:", threshold)

probs = np.zeros(len(X_test))
for model in models:
    probs += model.predict_proba(X_test)[:, 1]
probs /= len(models)

preds = (probs > threshold).astype(int)

submission = pd.DataFrame({
    "object_id": test["object_id"],
    "prediction": preds
})

os.makedirs("submissions", exist_ok=True)
submission.to_csv("submissions/submission_revert.csv", index=False)

print("Saved submissions/submission_revert.csv")
print("Positive predictions:", preds.sum())
