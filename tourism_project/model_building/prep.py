
import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# 1. Define paths
# ---------------------------------------------------------

DATA_PATH = "tourism_project/data/tourism.csv"
OUTPUT_DIR = "tourism_project/model_building"

TRAIN_PATH = os.path.join(OUTPUT_DIR, "train.csv")
TEST_PATH = os.path.join(OUTPUT_DIR, "test.csv")


# ---------------------------------------------------------
# 2. Load dataset from repository data folder
# ---------------------------------------------------------

print("========== DATA PREPARATION ==========")

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print("Original dataset shape:", df.shape)


# ---------------------------------------------------------
# 3. Remove unnecessary columns
# ---------------------------------------------------------

# Remove accidental CSV index column if present
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)
    print("Removed unnecessary column: Unnamed: 0")

# CustomerID is only an identifier and is not useful
# for predicting package purchase
if "CustomerID" in df.columns:
    df.drop(columns=["CustomerID"], inplace=True)
    print("Removed unnecessary column: CustomerID")


# ---------------------------------------------------------
# 4. Remove duplicate records
# ---------------------------------------------------------

duplicates = df.duplicated().sum()
print("\nDuplicate records found:", duplicates)

if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print("Duplicate records removed.")


# ---------------------------------------------------------
# 5. Display missing values
# ---------------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())


# ---------------------------------------------------------
# 6. Split cleaned data into train and test sets
# ---------------------------------------------------------

train_data, test_data = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["ProdTaken"]
)


# ---------------------------------------------------------
# 7. Save train and test datasets locally
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_data.to_csv(TRAIN_PATH, index=False)
test_data.to_csv(TEST_PATH, index=False)


# ---------------------------------------------------------
# 8. Print summary
# ---------------------------------------------------------

print("\n========== SPLIT SUMMARY ==========")

print("Cleaned dataset shape:", df.shape)
print("Training dataset shape:", train_data.shape)
print("Testing dataset shape:", test_data.shape)

print("\nTraining dataset saved to:", TRAIN_PATH)
print("Testing dataset saved to:", TEST_PATH)

print("\nData preparation completed successfully!")
