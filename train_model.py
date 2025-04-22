import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("Health Advisor.csv")
df.columns = df.columns.str.strip()

# Keep relevant columns and drop missing values
df = df[['Gender', 'Age', 'heart_rate', 'SpO2', 'Temprature', 'Health_status']].dropna()

# Encode Gender
gender_encoder = LabelEncoder()
df['Gender'] = gender_encoder.fit_transform(df['Gender'])

# Feature engineering
def categorize_heart_rate(hr):
    if hr < 60:
        return 'Low'
    elif 60 <= hr <= 100:
        return 'Normal'
    else:
        return 'High'

def categorize_spo2(spo2):
    return 'Low' if spo2 < 95 else 'Normal'

df['HR_Category'] = LabelEncoder().fit_transform(df['heart_rate'].apply(categorize_heart_rate))
df['SpO2_Status'] = LabelEncoder().fit_transform(df['SpO2'].apply(categorize_spo2))

# Simplify health status
def simplify_health_status(label):
    if label in [1, 2, 3]:
        return 'Healthy'
    elif label in [4, 5, 6]:
        return 'At Risk'
    else:
        return 'Unhealthy'

df['Health_status_grouped'] = df['Health_status'].apply(simplify_health_status)

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df['Health_status_grouped'])

# Feature matrix
X = df[['Gender', 'Age', 'heart_rate', 'SpO2', 'Temprature', 'HR_Category', 'SpO2_Status']]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# Save model
with open("health_advisor_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as health_advisor_model.pkl")
