import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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

datasets = {}
for path in file_paths:
    try:
        filename = path.split('/')[-1]
        delimiter = ';' if filename == "data.csv" else ','
        df = pd.read_csv(path, delimiter=delimiter)
        datasets[filename] = df
    except Exception as e:
        datasets[filename] = f"Error loading: {str(e)}"

# Fixed hidden layer size and updated hyperparameters
hidden_layer_sizes = [(64,)]
learning_rates = [0.001, 0.005]
activations = ['relu', 'tanh', 'logistic', 'identity']
solvers = ['adam', 'sgd']

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

        for hls, lr, act, solver in product(hidden_layer_sizes, learning_rates, activations, solvers):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = MLPClassifier(
                hidden_layer_sizes=hls,
                learning_rate_init=lr,
                activation=act,
                solver=solver,
                max_iter=500,
                random_state=42
            )

            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))

            if acc > best_acc:
                best_acc = acc
                best_params = {
                    "Hidden Layers": hls,
                    "Learning Rate": lr,
                    "Activation": act,
                    "Solver": solver
                }

        results.append({
            "Dataset": name,
            "Best Accuracy": best_acc,
            "Hidden Layers": best_params["Hidden Layers"],
            "Learning Rate": best_params["Learning Rate"],
            "Activation": best_params["Activation"],
            "Solver": best_params["Solver"]
        })

    except Exception as e:
        results.append({
            "Dataset": name,
            "Best Accuracy": "Error",
            "Hidden Layers": "-",
            "Learning Rate": "-",
            "Activation": "-",
            "Solver": "-"
        })

# Save results
results_df = pd.DataFrame(results)
output_dir = "output_MLP"
os.makedirs(output_dir, exist_ok=True)

results_path = os.path.join(output_dir, "mlp_results.csv")
results_df.to_csv(results_path, index=False)

# Plot results
plot_data = results_df[results_df["Best Accuracy"] != "Error"]
plt.figure(figsize=(10, 6))
plt.bar(plot_data["Dataset"], plot_data["Best Accuracy"])
plt.xticks(rotation=45, ha='right')
plt.ylabel("Accuracy")
plt.title("Best MLP Accuracy per Dataset (activation, solver, LR tuned)")
plt.tight_layout()
plot_path = os.path.join(output_dir, "mlp_accuracy_plot.png")
plt.savefig(plot_path)
plt.show()

print(f"Results saved to: {results_path}")
print(f"Plot saved to: {plot_path}")
