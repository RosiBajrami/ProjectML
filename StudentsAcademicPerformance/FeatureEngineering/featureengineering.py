import pandas as pd

# Load the preprocessed dataset
df = pd.read_csv("../Preprocessed_Dataset/preprocessed_data.csv")

# Create combined features
df["Total units enrolled"] = df["Curricular units 1st sem (enrolled)"] + df["Curricular units 2nd sem (enrolled)"]
df["Total units approved"] = df["Curricular units 1st sem (approved)"] + df["Curricular units 2nd sem (approved)"]
df["Total evaluations"] = df["Curricular units 1st sem (evaluations)"] + df["Curricular units 2nd sem (evaluations)"]
df["Total grade"] = df["Curricular units 1st sem (grade)"] + df["Curricular units 2nd sem (grade)"]

# Drop the original columns used in combination
columns_to_drop = [
    "Curricular units 1st sem (enrolled)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)"
]
df.drop(columns=columns_to_drop, inplace=True)

# Move the Target column to the end
target = df.pop("Target")
df["Target"] = target

# Save the final feature-engineered dataset
df.to_csv("../FeatureEngineered_Dataset/feature_engineered_data.csv", index=False)

print("Feature engineering complete. File saved with new features and old ones removed.")
