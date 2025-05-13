import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# Dataset paths
datasets = {
    "Original": "../Original_Dataset/data.csv",
    "Preprocessed": "../Preprocessed_Dataset/preprocessed_data.csv",
    "FeatureEngineered": "../FeatureEngineered_Dataset/feature_engineered_data.csv",
    "Pre_MinMax": "../Normalized_Datasets/preprocessed_minmax.csv",
    "Pre_ZScore": "../Normalized_Datasets/preprocessed_zscore.csv",
    "Pre_Decimal": "../Normalized_Datasets/preprocessed_decimal.csv",
    "FE_MinMax": "../Normalized_Datasets/fe_minmax.csv",
    "FE_ZScore": "../Normalized_Datasets/fe_zscore.csv",
    "FE_Decimal": "../Normalized_Datasets/fe_decimal.csv"
}

# Selected parameters to avoid too many combinations
criteria = ["gini", "entropy"]
n_estimators_list = [100, 200]
max_depths = [10, 20]

# Loop through datasets and run cleaner output
for name, path in datasets.items():
    for criterion in criteria:
        for n_estimators in n_estimators_list:
            for max_depth in max_depths:
                try:
                    df = pd.read_csv(path, sep=';' if name == "Original" else ',')
                    df.columns = df.columns.str.strip()

                    if "Target" not in df.columns:
                        raise ValueError(f"Target column missing. Actual columns: {df.columns.tolist()}")

                    X = df.drop("Target", axis=1)
                    y = df["Target"]

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        criterion=criterion,
                        max_depth=max_depth,
                        class_weight="balanced",
                        random_state=42
                    )

                    print(f"Running {name} dataset with criterion={criterion}, "
                          f"n_estimators={n_estimators}, max_depth={max_depth}")

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    print(f"Accuracy: {accuracy * 100:.2f}%")
                    print("---")

                except Exception as e:
                    print(f"{name:<15} ➤ ❌ Error: {e}")
