from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Encode SBERT embeddings
sbert_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
texts = hatespeech_df["Cleaned Text"].tolist()
text_embeddings = sbert_model.encode(texts, show_progress_bar=False)

# Prepare and scale handcrafted features
custom_features = hatespeech_df[["profanity_count", "profanity_score", "binary_profanity_match", "normalized_text_len"]].values
scaler = StandardScaler()
custom_features_scaled = scaler.fit_transform(custom_features.astype(np.float32))

# Combine SBERT + handcrafted features
X = np.hstack([np.array(text_embeddings).astype(np.float32), custom_features_scaled])
y = hatespeech_df["Hate or Non Hate speech"].values

# Balance classes using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Train LightGBM
clf = LGBMClassifier(
    n_estimators=150,
    learning_rate=0.05,
    num_leaves=64,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Evaluation
print(classification_report(y_test, y_pred, target_names=["Non-Hate", "Hate"]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Hate", "Hate"], yticklabels=["Non-Hate", "Hate"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# Feature importance
importances = clf.feature_importances_
plt.figure(figsize=(12, 4))
plt.bar(range(len(importances)), importances)
plt.title("Feature Importances (SBERT + Custom)")
plt.xlabel("Feature Index")
plt.ylabel("Importance Score")
plt.show()
