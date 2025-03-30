from fileinput import input
from flask import Flask, render_template, request
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

app = Flask(__name__)

#Training data
x = [[30], [40], [50], [60], [20], [10], [70]]
y = [0, 1, 1, 1, 0, 0, 1]

#train KNN model
knn_classifer = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
knn_classifer.fit(x, y)

#Train SVM model
svm_classifer = SVC(kernel='linear', random_state=0)
svm_classifer.fit(x, y)

@app.route('/',methods=['GET','POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            input_value = float(request.form['input_value'])
            algorithm = request.form['algorithm']
            x_test = [[input_value]]

            if algorithm == 'KNN':
                prediction = knn_classifer.predict(x_test)[0]

            elif algorithm == 'SVM':
                prediction = svm_classifer.predict(x_test)[0]

        except:
            prediction = "Invalid input"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
