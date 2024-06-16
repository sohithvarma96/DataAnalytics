
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


def predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    input_data = np.array(
        [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    input_data = input_data.reshape(1, -1)

    scaled_data = sc.transform(input_data)

    prediction_result = rf_model.predict(scaled_data)[0]  # rf_model was mentioned here right

    return prediction_result
predict (4,140,72,35,0,30.6,0.627,50)