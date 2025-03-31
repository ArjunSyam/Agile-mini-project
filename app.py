from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

#for ML models
from sklearn.neighbors import KNeighborsClassifier

#for dl models
from keras.models import load_model

#for QML models
from qiskit_aer import Aer
from qiskit.circuit import QuantumCircuit
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

app = Flask(__name__)

#load models
knn_classifer = joblib.load("models/knn_model.pkl")
dl_classifier = joblib.load("models/dl_model.pkl")
qml_classifier = joblib.load("models/qml_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route('/',methods=['GET','POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            headers = ['age', 'sex', 'cp', 'trestbps', 'chol',
                        'fbs', 'restecg', 'thalach','exang',
                        'oldpeak', 'slope', 'ca', 'thal']

            #adding all input values
            input_values = []
            input = []
            for i in range(len(headers)):
                input_values.append(float(request.form.get(headers[i])))

            #adds the row of input values
            input.append(input_values)

            #getting model type
            model = request.form.get("model")

            #printing values for crossreference
            '''
            for i in range(len(headers)):
                print(headers[i] , " : ", input_values[i])
            print("model : ", model)'''

            #coverting input data into dataframe

            print()
            print("input:", input)
            print("headers:", headers)
            print()

            X = pd.DataFrame(input, columns=headers, dtype=np.float32)
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            print()
            print("Dataframe: ")
            print(X)
            print()

            if model == "knn":
                print()
                print("KNN")

                prediction = knn_classifer.predict(X)

            elif model == "dl":
                print()
                print("Deep Learning")

                prediction = dl_classifier.predict(X)
                prediction = (prediction > 0.5)

            else:
                print()
                print("Quantum Machine Learning")

                prediction = qml_classifier.predict(X)

            prediction = int(prediction[0])
            print("prediction: " ,prediction)

        except Exception as e:
            prediction = "Invalid input"
            print("error: " ,e)

    return render_template('index.html', prediction=str(prediction))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
