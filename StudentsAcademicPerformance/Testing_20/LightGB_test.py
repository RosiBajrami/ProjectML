import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

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

# Apply SMOTE to balance training data
X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

# Train LightGBM Model
model = LGBMClassifier(
    n_estimators=600,
    learning_rate=0.02,
    max_depth=14,
    subsample=0.95,
    colsample_bytree=0.95,
    objective='multiclass',
    num_class=3,
    min_data_in_leaf=10,
    min_gain_to_split=0.001,
    verbosity=-1,
    random_state=42
)
model.fit(X_train, y_train)

# 1. Correlation with Target (Bar Plot)
target_corr = pd.concat([X_train, y_train], axis=1).corr()["Target"].drop("Target")
top_corr = target_corr.abs().sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 6))
top_corr.plot(kind='barh')
plt.title("Top 10 Features Correlated with Target")
plt.xlabel("Absolute Correlation")
plt.tight_layout()
plt.savefig("lgb_barplot.png")
plt.close()

# 2. Pairplot (Top 3 Features)
top3_features = top_corr.head(3).index.tolist()

# Train pairplot
train_df = pd.DataFrame(X_train, columns=X.columns)
train_df["Target"] = y_train.map({0: "Dropout", 1: "Enrolled", 2: "Graduate"})

sns.pairplot(train_df[top3_features + ["Target"]], hue="Target", palette="coolwarm")
plt.suptitle("Pairplot - Train Data", y=1.02)
plt.tight_layout()
plt.savefig("lgb_pairplot_train.png")
plt.close()

# Test pairplot (with predicted labels)
test_df = pd.DataFrame(X_test, columns=X.columns)
test_df["Target"] = model.predict(X_test)
test_df["Target"] = test_df["Target"].map({0: "Dropout", 1: "Enrolled", 2: "Graduate"})

sns.pairplot(test_df[top3_features + ["Target"]], hue="Target", palette="coolwarm")
plt.suptitle("Pairplot - Test Data (Predicted Labels)", y=1.02)
plt.tight_layout()
plt.savefig("lgb_pairplot_test.png")
plt.close()

# 3. Train vs Test Confusion Matrix
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

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
plt.savefig("lgb_confusion_matrix.png")
plt.close()
