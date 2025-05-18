import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import ConfusionMatrixDisplay

# Load Dataset
df = pd.read_csv("../Normalized_Datasets/fe_zscore.csv")
df.columns = df.columns.str.strip()


if "Target" not in df.columns:
    raise ValueError("Target column missing.")

X = df.drop("Target", axis=1)
y = df["Target"]
if y.dtype == object:
    y = y.map({'Dropout': 0, 'Enrolled': 1, 'Graduate': 2})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_train)

X_train_clustered = X_train.copy()
X_test_clustered = X_test.copy()
X_train_clustered["Cluster"] = kmeans.predict(X_train)
X_test_clustered["Cluster"] = kmeans.predict(X_test)

# Train RandomForest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42
)
rf.fit(X_train_clustered, y_train)

# Predict
y_train_pred = rf.predict(X_train_clustered)
y_test_pred = rf.predict(X_test_clustered)

# 1. Correlation with Target
target_corr = pd.concat([X_train_clustered, y_train], axis=1).corr()["Target"].drop("Target")
top_corr = target_corr.abs().sort_values(ascending=False).head(10)
top3_features = top_corr.head(3).index.tolist()

plt.figure(figsize=(8, 6))
top_corr.plot(kind='barh')
plt.title("Top 10 Features Correlated with Target")
plt.xlabel("Absolute Correlation")
plt.tight_layout()
plt.savefig("hybrid2_barplot.png")
plt.close()

# 2. Pairplot (Train)
train_df = X_train_clustered.copy()
train_df["Target"] = y_train.map({0: "Dropout", 1: "Enrolled", 2: "Graduate"})
sns.pairplot(train_df[top3_features + ["Target"]], hue="Target", palette="coolwarm")
plt.suptitle("Pairplot - Train Data", y=1.02)
plt.tight_layout()
plt.savefig("hybrid2_pairplot_train.png")
plt.close()

# 3. Pairplot (Test) 
test_df = X_test_clustered.copy()
test_df["Target"] = y_test_pred
test_df["Target"] = test_df["Target"].map({0: "Dropout", 1: "Enrolled", 2: "Graduate"})
sns.pairplot(test_df[top3_features + ["Target"]], hue="Target", palette="coolwarm")
plt.suptitle("Pairplot - Test Data (Predicted Labels)", y=1.02)
plt.tight_layout()
plt.savefig("hybrid2_pairplot_test.png")
plt.close()

# 4. Confusion Matrix Train vs Test
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(
    y_train, y_train_pred,
    display_labels=["Dropout", "Enrolled", "Graduate"],
    ax=axes[0],
    cmap="Greens"
)
axes[0].set_title("Train Confusion Matrix")

ConfusionMatrixDisplay.from_predictions(
    y_test, y_test_pred,
    display_labels=["Dropout", "Enrolled", "Graduate"],
    ax=axes[1],
    cmap="Blues"
)
axes[1].set_title("Test Confusion Matrix")

plt.tight_layout()
plt.savefig("hybrid2_confusion_matrix.png")
plt.close()
