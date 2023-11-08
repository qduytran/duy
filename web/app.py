from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = [int(request.form['age']), int(request.form['education']), bool(request.form['sex']), 
                float(request.form['cigsPerDay']), bool(request.form['BPMeds']), bool(request.form['prevalentStroke']), bool(request.form['prevalentHyp']),
                bool(request.form['diabetes']), float(request.form['totChol']), float(request.form['sysBP']), float(request.form['diaBP']), 
                float(request.form['BMI']), int(request.form['heartRate']), int(request.form['glucose'])]
        features = np.array(input_data).reshape(1, -1)
        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "Có nguy cơ mắc bệnh tim trong 10 năm tới."
        else:
            result = "Không có nguy cơ mắc bệnh tim trong 10 năm tới."

        return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8081)