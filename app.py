from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("health_advisor_model.pkl", "rb") as file:
    model = pickle.load(file)

# Gender and Category encoders (same as training)
gender_map = {"Male": 1, "Female": 0}
hr_category_map = {"Low": 0, "Normal": 1, "High": 2}
spo2_status_map = {"Low": 0, "Normal": 1}

def categorize_heart_rate(hr):
    if hr < 60:
        return "Low"
    elif 60 <= hr <= 100:
        return "Normal"
    else:
        return "High"

def categorize_spo2(spo2):
    return "Low" if spo2 < 95 else "Normal"

@app.route('/')
def home():
    return "Health Advisor Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract values
    gender = gender_map.get(data['Gender'], 0)
    age = data['Age']
    heart_rate = data['heart_rate']
    spo2 = data['SpO2']
    temp = data['Temprature']
    
    hr_category = hr_category_map[categorize_heart_rate(heart_rate)]
    spo2_status = spo2_status_map[categorize_spo2(spo2)]

    # Combine into feature vector
    input_features = np.array([[gender, age, heart_rate, spo2, temp, hr_category, spo2_status]])

    prediction = model.predict(input_features)[0]

    # Optional: Return readable prediction
    labels = ["At Risk", "Healthy", "Unhealthy"]
    return jsonify({
        "prediction": int(prediction),
        "label": labels[prediction]
    })

if __name__ == '__main__':
    app.run(debug=True)
