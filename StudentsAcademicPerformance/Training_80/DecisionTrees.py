import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Dataset paths (adjust if needed)
datasets = {
    "Original": "data.csv",
    "Preprocessed": "preprocessed_data.csv",
    "FeatureEngineered": "feature_engineered_data.csv",
    "Pre_MinMax": "preprocessed_minmax.csv",
    "Pre_ZScore": "preprocessed_zscore.csv",
    "Pre_Decimal": "preprocessed_decimal.csv",
    "FE_MinMax": "fe_minmax.csv",
    "FE_ZScore": "fe_zscore.csv",
    "FE_Decimal": "fe_decimal.csv"
}

# Hypertuned config — very fast to run
params = [
    {"max_depth": 6, "min_samples_leaf": 2},
    {"max_depth": 8, "min_samples_leaf": 3}
]

for name, path in datasets.items():
    for p in params:
        try:
            # Load dataset
            df = pd.read_csv(path, sep=';' if name == "Original" else ',')
            df.columns = df.columns.str.strip()

            if "Target" not in df.columns:
                raise ValueError(f"Target column missing. Columns: {df.columns.tolist()}")

            X = df.drop("Target", axis=1)
            y = df["Target"]
            if y.dtype == object:
                y = y.map({'Dropout': 0, 'Enrolled': 1, 'Graduate': 2})

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

            model = DecisionTreeClassifier(
                max_depth=p["max_depth"],
                min_samples_leaf=p["min_samples_leaf"],
                random_state=42
            )

            print(f"Running {name} dataset with max_depth={p['max_depth']}, "
                  f"min_samples_leaf={p['min_samples_leaf']}")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {acc * 100:.2f}%")
            print("---")

        except Exception as e:
            print(f"{name:<15} ➤  Error: {e}")
