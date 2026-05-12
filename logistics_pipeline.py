# logistics_pipeline.py
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import streamlit as st

# -----------------------------
# Step 1: Load JSON data
# -----------------------------
with open("synthetic_shipments.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data["shipments"])

# -----------------------------
# Step 2: Synthetic Feature Engineering
# -----------------------------
# Example: approximate distance between origin and destination (dummy values)
city_coords = {
    "Mumbai, India": (19.076, 72.8777),
    "Delhi, India": (28.7041, 77.1025),
    "Chennai, India": (13.0827, 80.2707),
    "Bangalore, India": (12.9716, 77.5946),
    "Kolkata, India": (22.5726, 88.3639),
    "Hyderabad, India": (17.3850, 78.4867),
    "Pune, India": (18.5204, 73.8567),
    "Ahmedabad, India": (23.0225, 72.5714),
    "Lucknow, India": (26.8467, 80.9462),
    "Jaipur, India": (26.9124, 75.7873)
}

def haversine(coord1, coord2):
    # Rough distance calculation in km
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

df["distance_km"] = df.apply(
    lambda row: haversine(city_coords.get(row["origin"], (0,0)),
                          city_coords.get(row["destination"], (0,0))),
    axis=1
)

# Time-to-delivery (hours) = difference between expected_delivery and last_update
df["expected_delivery"] = pd.to_datetime(df["expected_delivery"])
df["last_update"] = pd.to_datetime(df["last_update"])
df["time_to_delivery_hours"] = (df["expected_delivery"] - df["last_update"]).dt.total_seconds() / 3600

# -----------------------------
# Step 3: Encode categorical variables
# -----------------------------
features = ["origin", "destination", "status", "carrier", "weather_condition", "traffic_condition", "distance_km", "time_to_delivery_hours"]
target = "delay_risk"

label_encoders = {}
for col in ["origin", "destination", "status", "carrier", "weather_condition", "traffic_condition", target]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df[features]
y = df[target]

# -----------------------------
# Step 4: Pipeline Automation
# -----------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=500))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Step 5: Streamlit Dashboard
# -----------------------------
def run_dashboard():
    st.title("Shipment Delay Risk Prediction Dashboard")

    st.write("### Enter Shipment Details")
    origin = st.selectbox("Origin", df["origin"].unique())
    destination = st.selectbox("Destination", df["destination"].unique())
    status = st.selectbox("Status", df["status"].unique())
    carrier = st.selectbox("Carrier", df["carrier"].unique())
    weather = st.selectbox("Weather Condition", df["weather_condition"].unique())
    traffic = st.selectbox("Traffic Condition", df["traffic_condition"].unique())
    distance = st.number_input("Distance (km)", min_value=0.0, value=500.0)
    time_to_delivery = st.number_input("Time to Delivery (hours)", min_value=0.0, value=24.0)

    if st.button("Predict Delay Risk"):
        sample = pd.DataFrame([{
            "origin": origin,
            "destination": destination,
            "status": status,
            "carrier": carrier,
            "weather_condition": weather,
            "traffic_condition": traffic,
            "distance_km": distance,
            "time_to_delivery_hours": time_to_delivery
        }])

        prediction = pipeline.predict(sample)[0]
        risk_label = label_encoders["delay_risk"].inverse_transform([prediction])[0]
        st.success(f"Predicted Delay Risk: **{risk_label}**")

if __name__ == "__main__":
    run_dashboard()