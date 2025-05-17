import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

file_paths = [
    "../Original_Dataset/data.csv",
    "../Preprocessed_Dataset/preprocessed_data.csv",
    "../FeatureEngineered_Dataset/feature_engineered_data.csv",
    "../Normalized_Datasets/preprocessed_minmax.csv",
    "../Normalized_Datasets/preprocessed_zscore.csv",
    "../Normalized_Datasets/preprocessed_decimal.csv",
    "../Normalized_Datasets/fe_minmax.csv",
    "../Normalized_Datasets/fe_zscore.csv",
    "../Normalized_Datasets/fe_decimal.csv"
]

def detect_delimiter(filepath):
    with open(filepath, 'r') as file:
        first_line = file.readline()
        if first_line.count(';') > first_line.count(','):
            return ';'
        else:
            return ','

datasets = {}
for path in file_paths:
    try:
        delimiter = detect_delimiter(path)
        df = pd.read_csv(path, delimiter=delimiter)
        datasets[path.split('/')[-1]] = df
    except Exception as e:
        datasets[path.split('/')[-1]] = f"Error loading: {str(e)}"

C_values = [0.1, 1, 10]
kernels = ['linear', 'rbf']
gammas = ['scale', 'auto']

results = []

for name, df in datasets.items():
    if isinstance(df, str):
        continue

    try:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        mask = X.notnull().all(axis=1)
        X = X[mask]
        y = y[mask.values]

        best_acc = 0
        best_params = {}

        for c, k, g in product(C_values, kernels, gammas):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = SVC(C=c, kernel=k, gamma=g)
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))

            if acc > best_acc:
                best_acc = acc
                best_params = {
                    "C": c,
                    "Kernel": k,
                    "Gamma": g
                }

        results.append({
            "Dataset": name,
            "Best Accuracy": best_acc,
            "C": best_params["C"],
            "Kernel": best_params["Kernel"],
            "Gamma": best_params["Gamma"]
        })

    except Exception as e:
        results.append({
            "Dataset": name,
            "Best Accuracy": "Error",
            "C": "-",
            "Kernel": "-",
            "Gamma": "-"
        })

results_df = pd.DataFrame(results)

output_dir = "output_SVM"
os.makedirs(output_dir, exist_ok=True)
results_path = os.path.join(output_dir, "svm_results.csv")
results_df.to_csv(results_path, index=False)

plot_data = results_df[results_df["Best Accuracy"] != "Error"]
plt.figure(figsize=(10, 6))
plt.bar(plot_data["Dataset"], plot_data["Best Accuracy"])
plt.xticks(rotation=45, ha='right')
plt.ylabel("Accuracy")
plt.title("Best SVM Accuracy per Dataset (with Hyperparameter Tuning)")
plt.tight_layout()
plot_path = os.path.join(output_dir, "svm_accuracy_plot.png")
plt.savefig(plot_path)
plt.show()

print(f"Results saved to: {results_path}")
print(f"Plot saved to: {plot_path}")
