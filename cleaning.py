import pandas as pd
from datetime import datetime

# Load dataset
df = pd.read_csv("dataset.csv", sep="\t")

print("Dataset Loaded Successfully")

# Missing Values

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Fill missing Income values with median
df["Income"] = df["Income"].fillna(df["Income"].median())


# Remove Duplicates

duplicates_before = df.duplicated().sum()
print("\nDuplicate Rows Before Cleaning:", duplicates_before)

df.drop_duplicates(inplace=True)

duplicates_after = df.duplicated().sum()
print("Duplicate Rows After Cleaning:", duplicates_after)


# Convert Date Column


df["Dt_Customer"] = pd.to_datetime(
    df["Dt_Customer"],
    dayfirst=True
)
# Feature Engineering

current_year = datetime.now().year

# Customer Age
df["Age"] = current_year - df["Year_Birth"]

# Total Amount Spent
df["Total_Spent"] = (
    df["MntWines"]
    + df["MntFruits"]
    + df["MntMeatProducts"]
    + df["MntFishProducts"]
    + df["MntSweetProducts"]
    + df["MntGoldProds"]
)

# Total Children
df["Total_Children"] = (
    df["Kidhome"]
    + df["Teenhome"]
)

print("\nFeature Engineering Completed")

# Outlier Detection

Q1 = df["Income"].quantile(0.25)
Q3 = df["Income"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[
    (df["Income"] < lower_limit)
    | (df["Income"] > upper_limit)
]

print("\nNumber of Income Outliers:")
print(len(outliers))

# Data Validation

print("\nDataset Shape:")
print(df.shape)

print("\nRemaining Missing Values:")
print(df.isnull().sum().sum())

print("\nRemaining Duplicate Rows:")
print(df.duplicated().sum())

# Preview New Features

print("\nSample Records:")
print(
    df[
        [
            "Age",
            "Total_Spent",
            "Total_Children"
        ]
    ].head()
)

# Save Cleaned Dataset

df.to_csv(
    "cleaned_dataset.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")
print("File Name: cleaned_dataset.csv")