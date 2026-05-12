import streamlit as st

# Inside run_dashboard()
st.write("### Model Evaluation")

# Confusion Matrix
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoders["delay_risk"].classes_,
            yticklabels=label_encoders["delay_risk"].classes_, ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title("Confusion Matrix - Delay Risk")
st.pyplot(fig_cm)

# ROC Curve
fig_roc, ax_roc = plt.subplots()
for i, color in zip(range(n_classes), colors):
    ax_roc.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f"Class {label_encoders['delay_risk'].classes_[i]} (AUC = {roc_auc[i]:.2f})")
ax_roc.plot([0,1], [0,1], "k--", lw=2)
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curve - Multiclass Delay Risk")
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)