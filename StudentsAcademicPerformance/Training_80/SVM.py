import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

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
        return ';' if first_line.count(';') > first_line.count(',') else ','

# Load datasets
datasets = {}
for path in file_paths:
    try:
        delimiter = detect_delimiter(path)
        df = pd.read_csv(path, delimiter=delimiter)
        datasets[os.path.basename(path)] = df
    except Exception as e:
        datasets[os.path.basename(path)] = f"Error loading: {str(e)}"

# Hyperparameters
C_values = [0.1, 1, 10]
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
gammas = ['scale', 'auto']
degree_values = [0, 2, 3, 4]

results = []

for name, df in datasets.items():
    print(f"\n🔍 Processing: {name}")
    if isinstance(df, str):
        print(f"❌ Failed to load: {df}")
        continue

    try:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Encode categorical features
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y.astype(str))

        # Drop NaNs
        mask = X.notnull().all(axis=1)
        X = X[mask]
        y = y[mask.values]

        best_acc = 0
        best_params = None

        for c, k, g in product(C_values, kernels, gammas):
            degrees = degree_values if k == 'poly' else [None]
            for d in degrees:
                print(f"→ Trying C={c}, kernel={k}, gamma={g}, degree={d}")
                model = SVC(C=c, kernel=k, gamma=g, degree=d if d is not None else 3)
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model.fit(X_train, y_train)
                    acc = accuracy_score(y_test, model.predict(X_test))

                    if acc > best_acc:
                        best_acc = acc
                        best_params = {
                            "C": c,
                            "Kernel": k,
                            "Gamma": g,
                            "Degree": d if d is not None else "-"
                        }
                except Exception as e:
                    print(f"⚠️ Model failed: {e}")

        if best_params:
            results.append({
                "Dataset": name,
                "Best Accuracy": best_acc,
                "C": best_params["C"],
                "Kernel": best_params["Kernel"],
                "Gamma": best_params["Gamma"],
                "Degree": best_params["Degree"]
            })
        else:
            results.append({
                "Dataset": name,
                "Best Accuracy": "Error",
                "C": "-",
                "Kernel": "-",
                "Gamma": "-",
                "Degree": "-"
            })

    except Exception as e:
        print(f"❌ Exception processing {name}: {e}")
        results.append({
            "Dataset": name,
            "Best Accuracy": "Error",
            "C": "-",
            "Kernel": "-",
            "Gamma": "-",
            "Degree": "-"
        })

# Save and display results
results_df = pd.DataFrame(results)
print("\n✅ Final Results:")
print(results_df)

output_dir = "output_SVM"
os.makedirs(output_dir, exist_ok=True)
results_path = os.path.join(output_dir, "svm_results.csv")
results_df.to_csv(results_path, index=False)

# Plot
plot_data = results_df[results_df["Best Accuracy"] != "Error"]
plt.figure(figsize=(10, 6))
plt.bar(plot_data["Dataset"], plot_data["Best Accuracy"])
plt.xticks(rotation=45, ha='right')
plt.ylabel("Accuracy")
plt.title("Best SVM Accuracy per Dataset (All Kernels + Degrees for Poly)")
plt.tight_layout()
plot_path = os.path.join(output_dir, "svm_accuracy_plot.png")
plt.savefig(plot_path)
plt.show()

print(f" Results saved to: {results_path}")
print(f"Plot saved to: {plot_path}")
