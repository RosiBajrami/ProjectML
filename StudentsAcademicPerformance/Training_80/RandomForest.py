import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Dataset names and paths (uploaded files)
rf_datasets = {
    "data.csv": "data.csv",
    "preprocessed_minmax.csv": "preprocessed_minmax.csv",
    "preprocessed_zscore.csv": "preprocessed_zscore.csv",
    "preprocessed_decimal.csv": "preprocessed_decimal.csv",
    "feature_engineered_data.csv": "feature_engineered_data.csv",
    "fe_minmax.csv": "fe_minmax.csv",
    "fe_zscore.csv": "fe_zscore.csv",
    "fe_decimal.csv": "fe_decimal.csv"
}

rf_results = []
label_encoder = LabelEncoder()

# Loop through each dataset and train Random Forest
for name, path in rf_datasets.items():
    try:
        df = pd.read_csv(path, sep=';' if name == "data.csv" else ',')
        df.columns = df.columns.str.strip()

        if "Target" not in df.columns:
            raise ValueError(f"'Target' column missing in {name}")

        X = df.drop("Target", axis=1)
        y = df["Target"]

        if y.dtype == 'O':
            y = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        model = RandomForestClassifier(
            n_estimators=100,
            criterion="gini",
            max_depth=None,
            class_weight="balanced",
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        rf_results.append({
            "Dataset": name,
            "Best Accuracy": acc,
            "Model": "Random Forest",
            "Hyperparameters": "n_estimators=100, criterion=gini, class_weight=balanced"
        })

    except Exception as e:
        rf_results.append({
            "Dataset": name,
            "Best Accuracy": "Error",
            "Model": "Random Forest",
            "Hyperparameters": "n_estimators=100, criterion=gini, class_weight=balanced",
            "Error": str(e)
        })

# Create results DataFrame
rf_summary_df = pd.DataFrame(rf_results)

# Save as CSV
csv_path = "random_forest_summary_results.csv"
rf_summary_df.to_csv(csv_path, index=False)
print(f"✅ CSV saved as: {csv_path}")

# Filter valid rows for plotting
plot_data = rf_summary_df[rf_summary_df["Best Accuracy"] != "Error"].copy()
plot_data["Best Accuracy"] = pd.to_numeric(plot_data["Best Accuracy"])

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(plot_data["Dataset"], plot_data["Best Accuracy"], color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Accuracy")
plt.title("Random Forest Accuracy per Dataset")
plt.tight_layout()

# Save and show plot
plot_path = "random_forest_accuracy_plot.png"
plt.savefig(plot_path)
plt.show()

print(f"✅ Plot saved as: {plot_path}")

