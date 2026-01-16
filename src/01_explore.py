print("01_explore.py is running")
import pandas as pd
train = pd.read_csv("data/raw/train_log.csv")
test  = pd.read_csv("data/raw/test_log.csv")

print("=== BASIC SHAPES ===")
print("Train Rows:", train.shape[0])
print("Train columns:", train.shape[1])
print("Test rows :", test.shape[0])
print("Test Columns :", test.shape[1])

print("\n=== COLUMNS ===")
print(train.columns.tolist())

print("\n=== MISSING VALUES (TRAIN) ===")
print(train.isnull().sum())

print("\n=== MISSING VALUES (TEST) ===")
print(test.isnull().sum())

print("\n=== OBJECT COUNTS ===")
print("Unique train objects:", train["object_id"].nunique())
print("Unique test objects :", test["object_id"].nunique())

print("\n=== OBSERVATIONS PER OBJECT (TRAIN) ===")
obs_train = train.groupby("object_id").size()
print(obs_train.describe())

print("\n=== OBSERVATIONS PER OBJECT (TEST) ===")
obs_test = test.groupby("object_id").size()
print(obs_test.describe())

print("\n=== TARGET DISTRIBUTION ===")
print(train["target"].value_counts())
print("\nTarget ratio:")
print(train["target"].value_counts(normalize=True))

print("\n=== SANITY CHECKS ===")
assert train["object_id"].isnull().sum() == 0, "Null object_id in train!"
assert test["object_id"].isnull().sum() == 0, "Null object_id in test!"
assert train["target"].isnull().sum() == 0, "Null target labels!"

print ("All sanity checks passed")

print("\n=== SAMPLE ROWS ===")
print(train.head())

print("\nExploration complete.")