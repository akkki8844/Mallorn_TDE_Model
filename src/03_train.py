import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import joblib
import os

train = pd.read_csv("data/processed/train_features.csv")

X = train.drop(["object_id", "target"], axis=1)
y = train["target"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(X))
models = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    model = lgb.LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.04,
        num_leaves=64,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=42 + fold,
        n_jobs=-1
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="binary_logloss"
    )

    oof[va_idx] = model.predict_proba(X_va)[:, 1]
    models.append(model)

best_f1 = 0.0
best_threshold = 0.5

for t in np.linspace(0.005, 0.08, 200):
    preds = (oof > t).astype(int)
    f1 = f1_score(y, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

os.makedirs("models", exist_ok=True)

for i, m in enumerate(models):
    joblib.dump(m, f"models/lgbm_fold_{i}.pkl")

np.save("models/best_threshold.npy", best_threshold)

print("Best CV F1:", best_f1)
print("Best threshold:", best_threshold)
