from flask import Flask, request, render_template

import pickle

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecret'

scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('svm_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/prediction', methods=['GET', 'POST'])
def Prediction():
    prediction = -1
    heart_disease_risk = False
    if request.method == 'POST':
        pregs = int(request.form.get('pregs'))
        gluc = int(request.form.get('gluc'))
        bp = int(request.form.get('bp'))
        skin = int(request.form.get('skin'))
        insulin = float(request.form.get('insulin'))
        bmi = float(request.form.get('bmi'))
        func = float(request.form.get('func'))
        age = int(request.form.get('age'))

        input_features = [[pregs, gluc, bp, skin, insulin, bmi, func, age]]
        prediction = model.predict(scaler.transform(input_features))
        prediction = prediction[0]

        if insulin >= 26 or gluc >= 100 or bp >= 120:
            heart_disease_risk = True

    return render_template('prediction.html', prediction=prediction, heart_disease_risk = heart_disease_risk)


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    return render_template('prediction.html')


@app.route('/bmi', methods=['GET', 'POST'])
def bmi():
    bmi = -1
    if request.method == 'POST':
        weight = float(request.form.get('weight'))
        height = float(request.form.get('height'))
        bmi = round(weight / (height / 100) ** 2, 2)
    return render_template('bmi.html', bmi=bmi)

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

if __name__ == '__main__':
    app.run(debug=True)