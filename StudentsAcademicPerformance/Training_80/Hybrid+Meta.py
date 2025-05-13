import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
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

# Base models
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    criterion="entropy",
    class_weight="balanced",
    random_state=42
)

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42
)

meta_model = LogisticRegression(max_iter=1000)

for name, path in datasets.items():
    try:
        df = pd.read_csv(path, sep=';' if name == "Original" else ',')
        df.columns = df.columns.str.strip()

        if "Target" not in df.columns:
            raise ValueError(f"Target column missing. Found: {df.columns.tolist()}")

        X = df.drop("Target", axis=1)
        y = df["Target"]

        if y.dtype == object:
            y = y.map({'Dropout': 0, 'Enrolled': 1, 'Graduate': 2})

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

        stacking_model = StackingClassifier(
            estimators=[('rf', rf), ('xgb', xgb)],
            final_estimator=meta_model,
            passthrough=False,
            n_jobs=-1
        )

        print(f"Running stacked model on {name} dataset (RF + XGB → Logistic Regression)")
        stacking_model.fit(X_train, y_train)
        y_pred = stacking_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc * 100:.2f}%")
        print("---")

    except Exception as e:
        print(f"{name:<15} ➤ ❌ Error: {e}")
