import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
n_clusters_list = [2, 3]
rf_n_estimators = [100, 200]
rf_max_depths = [8, 12]
rf_min_samples_leaf = [1, 3]

for name, path in datasets.items():
    try:
        df = pd.read_csv(path, sep=';' if name == "Original" else ',')
        df.columns = df.columns.str.strip()

        if "Target" not in df.columns:
            raise ValueError(f"Target column missing in {name} dataset.")

        X = df.drop("Target", axis=1)
        y = df["Target"]
        if y.dtype == object:
            y = y.map({'Dropout': 0, 'Enrolled': 1, 'Graduate': 2})

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for n_clusters in n_clusters_list:
            for n_estimators in rf_n_estimators:
                for max_depth in rf_max_depths:
                    for min_leaf in rf_min_samples_leaf:
                        # Add KMeans cluster as an extra feature
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        kmeans.fit(X_train)
                        X_train_clustered = X_train.copy()
                        X_test_clustered = X_test.copy()
                        X_train_clustered["Cluster"] = kmeans.predict(X_train)
                        X_test_clustered["Cluster"] = kmeans.predict(X_test)

                        # Train RandomForest on augmented data
                        rf = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_leaf=min_leaf,
                            random_state=42
                        )
                        rf.fit(X_train_clustered, y_train)
                        y_pred = rf.predict(X_test_clustered)
                        acc = accuracy_score(y_test, y_pred)

                        print(f"Dataset: {name} | Clusters: {n_clusters} | Estimators: {n_estimators} | "
                              f"MaxDepth: {max_depth} | MinLeaf: {min_leaf} ➤ Accuracy: {acc * 100:.2f}%")
                        print("---")

    except Exception as e:
        print(f"❌ Error in {name}: {e}")
