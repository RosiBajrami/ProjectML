import pandas as pd
from xgboost import XGBClassifier
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

# Tuned parameter grid (limited to avoid overload)
n_estimators_list = [200, 300]
learning_rates = [0.05, 0.1]
max_depths = [6, 8]
subsamples = [0.8]
colsample_bytree_list = [0.8]

for name, path in datasets.items():
    for n_estimators in n_estimators_list:
        for lr in learning_rates:
            for max_depth in max_depths:
                for subsample in subsamples:
                    for colsample in colsample_bytree_list:
                        try:
                            df = pd.read_csv(path, sep=';' if name == "Original" else ',')
                            df.columns = df.columns.str.strip()

                            if "Target" not in df.columns:
                                raise ValueError(f"Target column missing. Columns: {df.columns.tolist()}")

                            X = df.drop("Target", axis=1)
                            y = df["Target"]
                            if y.dtype == object:
                                y = y.map({'Dropout': 0, 'Enrolled': 1, 'Graduate': 2})

                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

                            model = XGBClassifier(
                                n_estimators=n_estimators,
                                learning_rate=lr,
                                max_depth=max_depth,
                                subsample=subsample,
                                colsample_bytree=colsample,
                                eval_metric='mlogloss',
                                random_state=42
                            )

                            print(f"Running {name} dataset with n_estimators={n_estimators}, learning_rate={lr}, "
                                  f"max_depth={max_depth}, subsample={subsample}, colsample_bytree={colsample}")

                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            acc = accuracy_score(y_test, y_pred)
                            print(f"Accuracy: {acc * 100:.2f}%")
                            print("---")

                        except Exception as e:
                            print(f"{name:<15} ➤ ❌ Error: {e}")
