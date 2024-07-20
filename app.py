from flask import Flask, render_template, request
import numpy as np
import pickle


app = Flask(__name__)
model = pickle.load(open('ckd.pkl', 'rb'))

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        blood_pressure = float(request.form['blood_pressure'])
        specific_gravity = float(request.form['specific_gravity'])
        albumin = float(request.form['albumin'])
        sugar = float(request.form['sugar'])
        red_blood_cells = float(request.form['red_blood_cells'])
        pus_cell = float(request.form['pus_cell'])
        pus_cell_clumps = float(request.form['pus_cell_clumps'])
        bacteria = float(request.form['bacteria'])
        blood_glucose_random = float(request.form['blood_glucose_random'])
        blood_urea = float(request.form['blood_urea'])
        serum_creatinine = float(request.form['serum_creatinine'])
        sodium = float(request.form['sodium'])
        potassium = float(request.form['potassium'])
        haemoglobin = float(request.form['haemoglobin'])
        packed_cell_volume = float(request.form['packed_cell_volume'])
        white_blood_cell_count = float(request.form['white_blood_cell_count'])
        red_blood_cell_count = float(request.form['red_blood_cell_count'])
        hypertension = float(request.form['hypertension'])
        diabetes_mellitus = float(request.form['diabetes_mellitus'])
        coronary_artery_disease = float(request.form['coronary_artery_disease'])
        appetite = float(request.form['appetite'])
        peda_edema = float(request.form['peda_edema'])
        aanemia = float(request.form['aanemia'])

        values = np.array([[age, blood_pressure, specific_gravity, albumin, sugar,
       red_blood_cells, pus_cell, pus_cell_clumps, bacteria,
       blood_glucose_random, blood_urea, serum_creatinine, sodium,
       potassium, haemoglobin, packed_cell_volume,
       white_blood_cell_count, red_blood_cell_count, hypertension,
       diabetes_mellitus, coronary_artery_disease, appetite,
       peda_edema, aanemia]])
        prediction = model.predict(values)

        return render_template('result.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)

