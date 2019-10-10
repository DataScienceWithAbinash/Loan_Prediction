import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import dump,load

app = Flask(__name__)
model = load('Loan.joblib')

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/form')
def predict():
    return render_template('Form.html')


@app.route('/prediction')
def prediction():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0],2)
    return render_template('Result.html',Prediction= 'prediction should be{}'.format(output))

@app.route('/result')
def result():
    return render_template('Resultpage.html')


if __name__ == '__main__':
        app.run(debug=True)

