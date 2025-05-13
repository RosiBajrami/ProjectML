import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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

# Parameter sets to test
hidden_layer_options = [(100,), (128, 64)]
activations = ["relu", "tanh"]
alphas = [0.0005, 0.001]

for name, path in datasets.items():
    for hidden_layers in hidden_layer_options:
        for activation in activations:
            for alpha in alphas:
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

                    model = MLPClassifier(
                        hidden_layer_sizes=hidden_layers,
                        activation=activation,
                        alpha=alpha,
                        max_iter=500,
                        random_state=42
                    )

                    print(f"Running {name} dataset with hidden_layer_sizes={hidden_layers}, "
                          f"activation={activation}, alpha={alpha}")

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    print(f"Accuracy: {acc * 100:.2f}%")
                    print("---")

                except Exception as e:
                    print(f"{name:<15} ➤ ❌ Error: {e}")
