from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
app = Flask(__name__)

# Load the trained model
with open("rf_model.pkl", 'rb') as f:
    rf_model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
  

    input_data = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    input_data = input_data.reshape(1, -1)
    

    scaled_data = sc.transform(input_data)

    prediction_result = rf_model.predict(scaled_data)[0] # rf_model was mentioned here right
    

    return prediction_result

if __name__ == '__main__':
    app.run(debug=True)
prediction(4,140,72,35,0,30.6,0.627,50)
# but if i run apps.py it throws an error that sklearn