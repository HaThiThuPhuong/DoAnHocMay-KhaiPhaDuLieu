from flask import Flask, request, render_template, jsonify, send_file
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load các mô hình
id3_model = pickle.load(open('models/id3_model.pkl', 'rb'))
svm_model = pickle.load(open('models/svm_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
ann_model = load_model('models/ann_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file uploaded'})
    
    try:
        # Đọc file CSV
        data = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Error reading file: {str(e)}'})
    
    try:
        # Xử lý dữ liệu
        X = data.iloc[:, 1:-1] 
        X_scaled = scaler.transform(X)
    except Exception as e:
        return jsonify({'error': f'Error processing data: {str(e)}'})

    # Dự đoán kết quả
    preds = svm_model.predict(X_scaled)
    results = ["Động kinh" if pred == 1 else "Bình thường" for pred in preds]

    # Thêm cột Name và Result
    num_patients = len(data)
    patient_names = [f"Bệnh nhân {i+1}" for i in range(num_patients)]
    data.insert(0, 'Tên', patient_names)
    data.insert(1, 'Kết quả', results)

    # Ghi file kết quả
    output_file = 'output_with_results.csv'
    data.to_csv(output_file, index=False)
    return send_file(output_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
