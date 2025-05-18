import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Load Dataset
df = pd.read_csv("../Normalized_Datasets/fe_zscore.csv")
df.columns = df.columns.str.strip()

# Prepare Data
if "Target" not in df.columns:
    raise ValueError("Target column missing.")

X = df.drop("Target", axis=1)
y = df["Target"]
if y.dtype == object:
    y = y.map({'Dropout': 0, 'Enrolled': 1, 'Graduate': 2})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train MLP
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=42)
mlp.fit(X_train, y_train)

# MLP Probabilities
train_probs = mlp.predict_proba(X_train)
test_probs = mlp.predict_proba(X_test)

# Train SVM on MLP outputs
svm = SVC(kernel='rbf', C=10, gamma='scale')
svm.fit(train_probs, y_train)

# Predictions
y_train_pred = svm.predict(train_probs)
y_test_pred = svm.predict(test_probs)

# 1. Correlation with Target
target_corr = pd.concat([X_train, y_train], axis=1).corr()["Target"].drop("Target")
top_corr = target_corr.abs().sort_values(ascending=False).head(10)
top3_features = top_corr.head(3).index.tolist()

plt.figure(figsize=(8, 6))
top_corr.plot(kind='barh')
plt.title("Top 10 Features Correlated with Target")
plt.xlabel("Absolute Correlation")
plt.tight_layout()
plt.savefig("hybrid1_barplot.png")
plt.close()

# 2. Pairplot (Train Data)
train_df = X_train.copy()
train_df["Target"] = y_train.map({0: "Dropout", 1: "Enrolled", 2: "Graduate"})

sns.pairplot(train_df[top3_features + ["Target"]], hue="Target", palette="coolwarm")
plt.suptitle("Pairplot - Train Data", y=1.02)
plt.tight_layout()
plt.savefig("hybrid1_pairplot_train.png")
plt.close()

# 3. Pairplot (Test Data with Predicted Labels)
test_df = X_test.copy()
test_df["Target"] = y_test_pred
test_df["Target"] = test_df["Target"].map({0: "Dropout", 1: "Enrolled", 2: "Graduate"})

sns.pairplot(test_df[top3_features + ["Target"]], hue="Target", palette="coolwarm")
plt.suptitle("Pairplot - Test Data (Predicted Labels)", y=1.02)
plt.tight_layout()
plt.savefig("hybrid1_pairplot_test.png")
plt.close()

# 4. Confusion Matrix Side-by-Side
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
plt.savefig("hybrid1_confusion_matrix.png")
plt.close()
