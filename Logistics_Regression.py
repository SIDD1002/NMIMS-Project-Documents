import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# -----------------------------
# Step 1: Load JSON data
# -----------------------------
with open("synthetic_shipments.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data["shipments"])

# -----------------------------
# Step 2: Feature Engineering
# -----------------------------
# Select relevant features
features = ["origin", "destination", "status", "carrier", "weather_condition", "traffic_condition"]
target = "delay_risk"

# Encode categorical variables
label_encoders = {}
for col in features + [target]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df[features]
y = df[target]

# -----------------------------
# Step 3: Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Step 4: Logistic Regression Model
# -----------------------------
model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500)
model.fit(X_train, y_train)

# -----------------------------
# Step 5: Evaluation
# -----------------------------
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Step 6: Example Prediction
# -----------------------------
sample = pd.DataFrame([{
    "origin": label_encoders["origin"].transform(["Mumbai, India"])[0],
    "destination": label_encoders["destination"].transform(["Delhi, India"])[0],
    "status": label_encoders["status"].transform(["In Transit"])[0],
    "carrier": label_encoders["carrier"].transform(["BlueDart Logistics"])[0],
    "weather_condition": label_encoders["weather_condition"].transform(["Heavy Rain"])[0],
    "traffic_condition": label_encoders["traffic_condition"].transform(["Moderate"])[0]
}])

predicted_class = model.predict(sample)[0]
print("Predicted Delay Risk:", label_encoders["delay_risk"].inverse_transform([predicted_class])[0])