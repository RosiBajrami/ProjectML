import pandas as pd
from lightgbm import LGBMClassifier
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

# Parameters
n_estimators_list = [600, 700]
learning_rates = [0.02, 0.03]
max_depths = [14, 16]
subsamples = [0.95]
colsample_bytree_list = [0.95]


for name, path in datasets.items():
    for n_estimators in n_estimators_list:
        for lr in learning_rates:
            for max_depth in max_depths:
                for subsample in subsamples:
                    for colsample in colsample_bytree_list:
                        try:
                            df = pd.read_csv(path, sep=';' if name == "Original" else ',')

                            # Clean column names
                            df.columns = df.columns.str.strip().str.replace(' ', '_')

                            if "Target" not in df.columns:
                                raise ValueError(f"Target column missing. Columns: {df.columns.tolist()}")

                            X = df.drop("Target", axis=1)
                            y = df["Target"]
                            if y.dtype == object:
                                y = y.map({'Dropout': 0, 'Enrolled': 1, 'Graduate': 2})

                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42)

                            X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

                            model = LGBMClassifier(
                                n_estimators=n_estimators,
                                learning_rate=lr,
                                max_depth=max_depth,
                                subsample=subsample,
                                colsample_bytree=colsample,
                                objective='multiclass',
                                num_class=3,
                                random_state=42,
                                min_data_in_leaf=10,
                                min_gain_to_split=0.001,
                                verbosity=-1  # suppress warnings
                            )

                            print(f"Running {name} dataset with n_estimators={n_estimators}, "
                                  f"learning_rate={lr}, max_depth={max_depth}, subsample={subsample}, "
                                  f"colsample_bytree={colsample}")

                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            acc = accuracy_score(y_test, y_pred)
                            print(f"Accuracy: {acc * 100:.2f}%")
                            print("---")

                        except Exception as e:
                            print(f"{name:<15} ➤ ❌ Error: {e}")
