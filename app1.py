from flask import Flask, request, render_template, jsonify, send_file
# from: Chỉ định rằng ta muốn lấy các thành phần cụ thể từ thư viện.
# flask: Tên thư viện dùng để tạo ứng dụng web.
# import: Nhập các thành phần sau vào chương trình.
# Flask: Đối tượng chính để tạo ứng dụng Flask.
# request: Dùng để lấy dữ liệu từ HTTP request (dữ liệu từ form, file tải lên, v.v.).
# render_template: Dùng để hiển thị trang web từ file HTML.
# jsonify: Chuyển đổi dữ liệu Python sang JSON để phản hồi cho client.
# send_file: Gửi file về cho người dùng tải xuống.
import pandas as pd
# import: Nhập thư viện pandas, dùng để xử lý dữ liệu dạng bảng.
# as pd: Đặt tên viết tắt pd để gọi thư viện nhanh hơn.
import numpy as np
# numpy: Hỗ trợ tính toán số học và xử lý mảng nhiều chiều.
# as np: Viết tắt để sử dụng nhanh hơn.
import pickle
# pickle → Tải mô hình ID3, SVM, Scaler đã lưu.
from tensorflow.keras.models import load_model
# tensorflow.keras.models: Module con của TensorFlow hỗ trợ deep learning.
# import load_model: Hàm dùng để tải mô hình mạng nơ-ron nhân tạo (.h5).

# 1. TẠO ỨNG DỤNG FLASK & TẢI MÔ HÌNH
app = Flask(__name__)
# Tạo một ứng dụng Flask và gán vào biến app.
# Flask(name)
# Flask: Đối tượng chính để tạo ứng dụng web.
# __name__: Biến đặc biệt trong Python, xác định tên module hiện tại.
# Khi chạy file trực tiếp, __name__ == "__main__", nếu file được import vào nơi khác, __name__ sẽ là tên file.

# Load các mô hình
id3_model = pickle.load(open('models/id3_model.pkl', 'rb'))
# id3_model.pkl: Mô hình cây quyết định ID3.
# pickle.load(open(..., 'rb'))
# open(...): Mở file mô hình.
# 'rb': Chế độ đọc file ở dạng nhị phân (r = read, b = binary).
# pickle.load(...): Giải mã nội dung file, khôi phục mô hình đã lưu.
svm_model = pickle.load(open('models/svm_model.pkl', 'rb'))
# svm_model.pkl: Mô hình Support Vector Machine (SVM).
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
# scaler.pkl: Bộ chuẩn hóa dữ liệu (chuẩn hóa đầu vào).
ann_model = load_model('models/ann_model.h5')
# load_model(...): Tải mô hình mạng nơ-ron nhân tạo từ file .h5.

# 2. XÂY DỰNG ROUTE (GIAO DIỆN & API DỰ ĐOÁN)
# 2.1. Route hiển thị trang web
@app.route('/')
# @: Ký hiệu decorator trong Python.
# app.route('/'): Xác định đường dẫn / (trang chủ).
def index(): # Hàm xử lý khi người dùng truy cập /.
    return render_template('index.html') # render_template(...): Trả về trang HTML để hiển thị.
# 2.2. Route nhận file CSV và dự đoán
@app.route('/predict', methods=['POST'])
# @:
# Trong Python, @ là ký hiệu dùng để khai báo decorator.
# Decorator là một hàm bọc lấy một hàm khác, giúp thay đổi hoặc mở rộng hành vi của hàm đó mà không sửa đổi mã nguồn gốc.
# app.route(...):
# app là đối tượng Flask đã khởi tạo trước đó (app = Flask(__name__)).
# .route(...) dùng để đăng ký một đường dẫn (route) cho ứng dụng Flask.
# Khi có yêu cầu gửi đến /predict, Flask sẽ gọi hàm tương ứng (predict).
# '/predict':
# Đây là đường dẫn (endpoint) được định nghĩa.
# Khi người dùng truy cập http://localhost:5000/predict, Flask sẽ chạy hàm predict().
# methods xác định loại HTTP request mà route này chấp nhận.
# 'POST' nghĩa là chỉ cho phép request kiểu POST (thường dùng để gửi dữ liệu từ client lên server).
# Nếu người dùng gửi request GET, Flask sẽ trả về lỗi 405 Method Not Allowed.
def predict(): # Hàm xử lý dữ liệu và trả kết quả dự đoán.
    # def: Từ khóa khai báo hàm trong Python.
    # predict: Tên hàm, có thể đặt bất kỳ, nhưng thường đặt theo chức năng của nó (ở đây là dự đoán).
    # XỬ LÝ DỮ LIỆU & DỰ ĐOÁN
    file = request.files['file']
    # request:
    #      Đây là một đối tượng Flask cung cấp thông tin về HTTP request hiện tại.
    #      Flask sẽ tự động tạo đối tượng request khi một yêu cầu HTTP được gửi đến server.
    # .files:
    #      Đây là một dictionary (từ điển) chứa các file mà client đã tải lên.
    #      Nếu không có file nào được tải lên, request.files sẽ rỗng {}.
    # ['file']:
    #      Lấy file có tên "file" từ request.files.
    #      Nếu client không gửi file nào hoặc gửi file với tên khác, Flask sẽ báo lỗi KeyError.
    if not file: # Kiểm tra nếu không có file nào được tải lên.
        return jsonify({'error': 'No file uploaded'}) # Trả về JSON chứa lỗi
    # Nếu không có file, trả về JSON chứa thông báo lỗi.
    # jsonify(...): Chuyển đổi dictionary Python thành JSON.
    # Flask sẽ tự động gửi HTTP response với Content-Type: application/json.
    try:
        # try:
        #     try là một khối mã để thử chạy lệnh có thể gây lỗi.
        #     Nếu lệnh trong try chạy bình thường, chương trình tiếp tục.
        #     Nếu có lỗi, chương trình chuyển ngay xuống except.
        # Đọc file CSV
        data = pd.read_csv(file) # Đọc file CSV vào DataFrame.
        # pd:
        #    Là alias của pandas (import pandas as pd).
        #    pandas là thư viện giúp xử lý dữ liệu bảng (giống như Excel).
        # .read_csv(file):
        #    Đọc file CSV và chuyển thành DataFrame (một dạng bảng dữ liệu của Pandas).
        #    file: Đây là file đã nhận từ request.files['file'].
    except Exception as e: # Nếu xảy ra lỗi, trả về JSON thông báo lỗi.
        # Nếu pd.read_csv(file) gặp lỗi (file sai định dạng, bị hỏng, không phải CSV), khối này sẽ bắt lỗi.
        # return jsonify({'error': f'Error reading file: {str(e)}'})
        # Trả về lỗi dưới dạng JSON.
        # f'Error reading file: {str(e)}':
        #     f'': F-string giúp nhúng biến vào chuỗi.
        #     {str(e)}: Chuyển lỗi thành chuỗi để hiển thị.
    
    try:
        # Xử lý dữ liệu
        X = data.iloc[:, 1:-1] 
        # data.iloc[:, 1:-1]
        # .iloc: Là thuộc tính của Pandas DataFrame, dùng để truy xuất dữ liệu theo chỉ số hàng, cột.
        # [:, 1:-1]:
        #      : → Lấy tất cả các dòng.
        #      1:-1 → Lấy các cột từ cột thứ 2 đến cột áp chót (không lấy cột đầu tiên và cuối cùng).
        X_scaled = scaler.transform(X)
        scaler.transform(X)
            # scaler:
            #     Đã được tải từ file scaler.pkl.
            #     Là bộ chuẩn hóa dữ liệu giúp đưa giá trị về cùng một khoảng (ví dụ: 0 đến 1 hoặc -1 đến 1).
            # .transform(X):
            #       Chuyển đổi dữ liệu X bằng cách áp dụng chuẩn hóa.
            #       Giúp mô hình dự đoán chính xác hơn.
    except Exception as e:
        return jsonify({'error': f'Error processing data: {str(e)}'})

    # Dự đoán kết quả
    preds = svm_model.predict(X_scaled)
    # svm_model.predict(X_scaled)
    #     svm_model: Đã được tải từ svm_model.pkl.
    #     .predict(X_scaled):
    #          Dùng mô hình SVM để dự đoán nhãn bệnh cho từng dòng dữ liệu.
    #          Trả về danh sách các số:
    #              1: Bệnh nhân bị động kinh.
    #              0: Bệnh nhân bình thường.
    results = ["Động kinh" if pred == 1 else "Bình thường" for pred in preds]
    # Đây là một list comprehension, giúp tạo danh sách nhanh hơn vòng lặp for.
    # Nếu pred == 1, gán "Động kinh", nếu không gán "Bình thường".

    # Thêm cột Name và Result
    num_patients = len(data)
    patient_names = [f"Bệnh nhân {i+1}" for i in range(num_patients)]
    data.insert(0, 'Tên', patient_names)
    data.insert(1, 'Kết quả', results)

    # Ghi file kết quả
    output_file = 'output_with_results.csv' # Tên file CSV sẽ được tạo ra.
    data.to_csv(output_file, index=False)
    # .to_csv(...): Xuất DataFrame data thành file CSV.
    # index=False: Không lưu chỉ số dòng vào file.
    return send_file(output_file, as_attachment=True)
    # send_file(...): Gửi file về cho client.
    # as_attachment=True: Buộc trình duyệt tải file về thay vì hiển thị trực tiếp.

if __name__ == '__main__':
    # Đây là một điều kiện kiểm tra xem tệp Python hiện tại có đang được chạy trực tiếp hay không.
    # Khi bạn chạy tệp này (python your_script.py), biến __name__ sẽ có giá trị là '__main__', nên khối lệnh bên trong sẽ được thực thi.
    # Nếu tệp này được nhập (import) vào một tệp khác, __name__ sẽ không phải là '__main__', nên đoạn mã này sẽ không chạy.
    app.run(debug=True)
    # app.run() sẽ khởi động ứng dụng Flask và chạy một server nội bộ để bạn có thể truy cập nó thông qua trình duyệt hoặc API client như Postman.
    # debug=True bật chế độ debug:
    #      Tự động tải lại khi bạn chỉnh sửa code (Auto-reload).
    #      Hiển thị lỗi chi tiết trên giao diện web nếu có lỗi xảy ra.
