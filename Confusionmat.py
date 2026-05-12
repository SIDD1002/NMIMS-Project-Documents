import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoders["delay_risk"].classes_,
            yticklabels=label_encoders["delay_risk"].classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Delay Risk")
plt.show()

# -----------------------------
# ROC Curve (Multiclass)
# -----------------------------
# Binarize labels for one-vs-rest ROC
y_test_bin = label_binarize(y_test, classes=[0,1,2])  # assuming 0=Low,1=Medium,2=High
y_score = pipeline.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_test_bin.shape[1]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(7,6))
colors = ["blue", "orange", "green"]
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"Class {label_encoders['delay_risk'].classes_[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0,1], [0,1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Multiclass Delay Risk")
plt.legend(loc="lower right")
plt.show()