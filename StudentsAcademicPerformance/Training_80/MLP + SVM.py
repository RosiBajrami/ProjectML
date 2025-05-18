import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

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

param_combinations = [
    {'mlp_hidden': (100,), 'mlp_lr': 0.001, 'svm_c': 1, 'svm_kernel': 'rbf'},
    {'mlp_hidden': (100,), 'mlp_lr': 0.01, 'svm_c': 1, 'svm_kernel': 'rbf'},
    {'mlp_hidden': (50,), 'mlp_lr': 0.001, 'svm_c': 0.5, 'svm_kernel': 'linear'},
    {'mlp_hidden': (150,), 'mlp_lr': 0.005, 'svm_c': 1.5, 'svm_kernel': 'poly'},
    {'mlp_hidden': (80, 40), 'mlp_lr': 0.001, 'svm_c': 2, 'svm_kernel': 'rbf'},
    {'mlp_hidden': (120,), 'mlp_lr': 0.0005, 'svm_c': 1, 'svm_kernel': 'linear'},
    {'mlp_hidden': (100, 50), 'mlp_lr': 0.002, 'svm_c': 2, 'svm_kernel': 'poly'},
    {'mlp_hidden': (60,), 'mlp_lr': 0.001, 'svm_c': 0.8, 'svm_kernel': 'rbf'}
]

for dataset_name, path in datasets.items():
    try:
        df = pd.read_csv(path, sep=';' if dataset_name == "Original" else ',')
        df.columns = df.columns.str.strip()
        if "Target" not in df.columns:
            raise ValueError(f"Missing 'Target' column in {dataset_name}")
        X = df.drop("Target", axis=1)
        y = df["Target"]
        if y.dtype == object:
            y = y.map({'Dropout': 0, 'Enrolled': 1, 'Graduate': 2})
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"\nRunning MLP+SVM on {dataset_name} dataset...")
        for idx, params in enumerate(param_combinations, 1):
            mlp = MLPClassifier(hidden_layer_sizes=params['mlp_hidden'],
                                learning_rate_init=params['mlp_lr'],
                                max_iter=200,
                                early_stopping=True,
                                n_iter_no_change=10,
                                tol=1e-4,
                                random_state=42)
            mlp.fit(X_train, y_train)
            mlp_features = mlp.predict_proba(X_train)
            svm = SVC(C=params['svm_c'], kernel=params['svm_kernel'], probability=False, random_state=42)
            svm.fit(mlp_features, y_train)
            test_features = mlp.predict_proba(X_test)
            y_pred = svm.predict(test_features)
            acc = accuracy_score(y_test, y_pred)
            print(f"  Combo {idx}: Accuracy = {acc * 100:.2f}% "
                  f"(MLP={params['mlp_hidden']}, LR={params['mlp_lr']}, SVM_C={params['svm_c']}, Kernel={params['svm_kernel']})")
    except Exception as e:
        print(f"Error in {dataset_name}: {e}")