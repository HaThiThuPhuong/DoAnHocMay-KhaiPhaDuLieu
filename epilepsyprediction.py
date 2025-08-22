
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Tiền xử lý dữ liệu
data = pd.read_csv('Epileptic Seizure Recognition.csv')
print("data.info()",data.info())
print(data.describe())
# Xóa dữ liệu bị thiếu hoặc trùng lặp 
data.dropna(inplace=True)  
data.drop_duplicates(inplace=True)
print("data.info()",data.info())
print(data.describe())
# Tách dữ liệu thành X( đặc trưng) và Y(nhãnnhãn)
X = data.iloc[:, 1:-1]  
y = data.iloc[:, -1].apply(lambda x: 1 if x == 1 else 0)  

print (y)
print("Original class distribution:")
print(y.value_counts())

# 2. Chuẩn hóa và cân bằng dữ liệu
# 2.1. Chuẩn hóa dữ liệuliệu
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Chia dữ liệu thành train-testtest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2.2. Cân bằng dữ liệu bằng smote
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("After SMOTE class distribution:")
print(pd.Series(y_train_smote).value_counts())

# 2.3. Vẽ biểu đồ đếm dữ liệu 
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train_smote, hue=y_train_smote, palette="viridis", dodge=False, legend=False)
plt.title('Distribution of Training Data After SMOTE')
plt.xlabel('Classes')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['No Seizure', 'Seizure'])

# 3. Huấn luyện và đánh giá mô hình 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 3.1. Huấn luyện mô hình id3
id3 = DecisionTreeClassifier(criterion='entropy', max_depth=15, random_state=42)
id3.fit(X_train_smote, y_train_smote)
# Dự đoán với mô hinhf id3
y_pred_id3 = id3.predict(X_test)
y_pred_prob_id3 = id3.predict_proba(X_test)[:, 1]
# Đánh giá mô hình id3 
print("ID3 Accuracy:", accuracy_score(y_test, y_pred_id3))
print(classification_report(y_test, y_pred_id3))

# 3.2. Huấn luyện mô hình SVM
svm = SVC(
    C=1.0,                  
    kernel='rbf',           
    gamma='scale',           
    probability=True,     
    random_state=42      
)
# Dự đoán với mô hình SVM
svm.fit(X_train_smote, y_train_smote)
y_pred_svm = svm.predict(X_test)
y_pred_prob_svm = svm.predict_proba(X_test)[:, 1] 
# Đánh giá mô hình SVM
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# 3.3. Huấn luyện mô hình noron nhân tạo(ANN)
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  
])
# Biên dịch mô hình 
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
# Huấn luyện mô hình 
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
# Đánh giá mô hình
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)

y_pred_ann = model.predict(X_test).flatten()
y_pred_ann_labels = np.round(y_pred_ann).astype(int)  

# 4. Vẽ ma trận nhầm lẫn 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# Định nghĩa hàm vẽ ma trận nhầm lẫn 
def plot_confusion_matrix(y_true, y_pred, model_name):
    # Tính ma trận nhầm lẫn 
    conf_matrix = confusion_matrix(y_true, y_pred)
    # Tạo biểu đồ mới 
    plt.figure(figsize=(6, 5))
    # Vẽ ma trận nhầm lẫn dưới dạng heatmap
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Seizure', 'Seizure'], 
                yticklabels=['No Seizure', 'Seizure'])
    # Thêm tiêu đề và nhãn trục 
    plt.title(f'Confusion Matrix ({model_name})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
  
# Vẽ ma trận nhầm lẫn cho các mô hình 
plot_confusion_matrix(y_test, y_pred_id3, "ID3")
plot_confusion_matrix(y_test, y_pred_svm, "SVM")
plot_confusion_matrix(y_test, y_pred_ann_labels, "ANN")

# 6. Lưu mô hình vào file 
import os
import pickle
# Tạo thư mục chứa mô hình 
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

# Lưu mô hình ID3
with open(os.path.join(models_dir, 'id3_model.pkl'), 'wb') as file:
    pickle.dump(id3, file)

# Lưu mô hình SVM
with open(os.path.join(models_dir, 'svm_model.pkl'), 'wb') as file:
    pickle.dump(svm, file)

# Lưu scaler
with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as file:
    pickle.dump(scaler, file)

# Lưu mô hình ANN
model.save(os.path.join(models_dir, 'ann_model.h5'))

print("Mô hình được lưu trữ thành công")

plt.show()