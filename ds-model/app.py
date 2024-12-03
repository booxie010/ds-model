import pandas as pd
import pickle

from flask import Flask, request, jsonify
from flask_cors import CORS #import Flask-CORS

#Membuat aplikasi Flask
app = Flask(__name__)

#Mengaktifkan CORS 
CORS(app) #untuk mengizinkan semua domain

#Memuat model yang sudah disimpan
with open('./ds-model/model.pkl', 'rb') as file:
    model = pickle.load(file)

#Endpoint untuk halaman home
@app.route('/')
def welcome():
    return "<h1>Selamat Datang di API DS Model</h1>"

#Endpoint untuk prediksi diabetes
@app.route('/predict', methods=['POST'])
def predict_diabetes():
    #Gunakan try catch untuk menampilkan status gangguan atau error
    try:
        #Mengambil data dari request
        data = request.get_json()

        #input untuk memprediksi
        #Prediksi diabetes berdasarkan faktor"
        input_data = pd.DataFrame([{
            "Pregnancies": data['Pregnancies'],
            "Glucose": data['Glucose'],
            "BloodPressure": data['BloodPressure'],
            "SkinThickness": data['SkinThickness'],
            "Insulin": data['Insulin'],
            "BMI": data['BMI'],
            "DiabetesPedigreeFunction": data['DiabetesPedigreeFunction'],
            "Age": data['Age']
        }])

        #Melakukan prediksi
        prediction = model.predict(input_data)

        probabilities = model.predict_proba(input_data)

        #probabilitas positif dan negatif dalam bentuk persentase
        probability_negative = probabilities[0][0] * 100; #persentase untuk kelas 0 negatif
        probability_positive = probabilities[0][1] * 100; #persentase untuk kelas 1 positif
        

        #prediksi output (untuk 1 itu positif diabetes)
        if prediction[0] == 1:
            result = f'Anda memiliki peluang menderita diabetes berdasarkan model KNN kami, Kemungkinan menderita diabetes adalah {probability_positive:.2f}%'
        else:
            result = "Hasil prediksi menunjuukan anda kemungkinan rendah diabetes"

        #menampilkan hasil prediksi dan probabilitas dalam bentuk json
        return jsonify({
            'prediction': result,
            'probabilities':{
                'negative': f"{probability_negative:.2f}%",
                'positive': f"{probability_positive:.2f}%"
            }
            })

    except Exception as e:
        return jsonify({'error': str(e)}),400
    

if __name__ == "__main__":
    app.run(debug=True)