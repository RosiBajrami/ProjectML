import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("../Original_Dataset/data.csv", sep=';')

# Drop unnecessary columns
columns_to_drop = [
    "Curricular units 1st sem (credited)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (without evaluations)",
    "Nacionality",
    "Educational special needs",
    "International",
    "Application order",
    "Previous qualification",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
    "Course"
]
df = df.drop(columns=columns_to_drop)

# Label encode the target column
label_encoder = LabelEncoder()
df["Target"] = label_encoder.fit_transform(df["Target"])

# One-hot encode selected categorical columns
categorical_columns = ['Gender', 'Application mode', 'Marital status']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Convert any boolean columns to integers (0 and 1)
bool_columns = df.select_dtypes(include='bool').columns
df[bool_columns] = df[bool_columns].astype(int)

# Move the Target column to the end
target = df.pop("Target")
df["Target"] = target

# Save the final preprocessed dataset
df.to_csv("../Preprocessed_Dataset/preprocessed_data.csv", index=False, header=True)
