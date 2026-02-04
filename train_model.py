import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

data = pd.read_csv("cybercrime.csv")
data = data.dropna()

# Create encoders
encoders = {}
columns = ["City","Crime_Type","Time_of_Crime","Victim_Age_Group",
           "Transaction_Mode","Bank_Type","Day_of_Week","Location"]

for col in columns:
    encoders[col] = LabelEncoder()
    data[col] = encoders[col].fit_transform(data[col])

X = data[["City","Crime_Type","Amount","Time_of_Crime",
          "Victim_Age_Group","Transaction_Mode","Bank_Type","Day_of_Week"]]

y = data["Location"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train,y_train)

print("Accuracy:", model.score(X_test,y_test))

joblib.dump(model,"cybercrime_model.pkl")

# Save encoders
for key in encoders:
    joblib.dump(encoders[key],f"{key}_encoder.pkl")
