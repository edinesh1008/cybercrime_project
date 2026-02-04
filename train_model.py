import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
data = pd.read_csv("cybercrime.csv")

# Remove missing values
data = data.dropna()

# Create encoders
city_encoder = LabelEncoder()
crime_encoder = LabelEncoder()
location_encoder = LabelEncoder()

data["City"] = city_encoder.fit_transform(data["City"])
data["Crime_Type"] = crime_encoder.fit_transform(data["Crime_Type"])
data["Location"] = location_encoder.fit_transform(data["Location"])

# Features & Target
X = data[["City","Crime_Type","Amount"]]
y = data["Location"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Improved Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save model & encoders
joblib.dump(model, "cybercrime_model.pkl")
joblib.dump(city_encoder, "city_encoder.pkl")
joblib.dump(crime_encoder, "crime_encoder.pkl")
joblib.dump(location_encoder, "location_encoder.pkl")