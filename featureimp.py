import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Feature Importance
# -----------------------------
# Get coefficients from logistic regression
coefficients = pipeline.named_steps["logreg"].coef_

# For multinomial logistic regression, we’ll average absolute values across classes
importance = abs(coefficients).mean(axis=0)

feature_names = X.columns
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(8,6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
plt.title("Feature Importance - Logistic Regression")
plt.xlabel("Average Coefficient Magnitude")
plt.ylabel("Feature")
plt.show()