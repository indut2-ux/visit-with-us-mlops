
import pandas as pd
import sys

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
]

print("=" * 60)
print("DATA VALIDATION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)

missing_columns = [
    column for column in EXPECTED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    print("Validation Failed!")
    print("Missing columns:", missing_columns)
    sys.exit(1)

print("\nValidation Successful!")
print("All expected columns are present.")

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["ProdTaken"].value_counts())

print("\nData validation completed successfully!")
