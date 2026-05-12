st.write("### Feature Importance")

fig_imp, ax_imp = plt.subplots()
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis", ax=ax_imp)
ax_imp.set_title("Feature Importance - Logistic Regression")
ax_imp.set_xlabel("Average Coefficient Magnitude")
ax_imp.set_ylabel("Feature")
st.pyplot(fig_imp)