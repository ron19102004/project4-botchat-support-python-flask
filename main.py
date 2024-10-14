import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from data import get_answer_data,get_training_data
from flask import Flask, request, jsonify

training_data = get_training_data()
answer_data = get_answer_data()

app = Flask(__name__)

# punkt_tab để tách văn bản
nltk.download('punkt')

# từ dừng tuỳ chỉnh
custom_stopwords = [
    "và", "là", "của", "trong", "không", "để", "một", "theo", "tại", "nhưng", 
    "thì", "đó", "lại", "hơn", "chỉ", "sẽ", "với", "nếu", "đang", "như", "thế",
    "khi", "vì", "có thể", "cũng", "đã", "hay", "mà", "mình", "ai", "đều",
    "làm", "bởi", "khác", "còn", "đến", "họ", "sau", "trước", "bên", "được",
    "tất cả", "đâu", "khoảng", "rất", "nhất", "thêm", "thể", "nhiều", "sao", 
    "vậy", "hầu hết", "bất cứ", "người", "thời gian", "cả", "chừng", "số", 
    "vô cùng", "gì", "từng", "thực sự", "bây giờ", "trong khi", "tại sao",
    "có lẽ", "ngày"
]

# tiền xử lý đầu vào
def preprocess(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]  # Chỉ giữ lại chữ cái
    tokens = [word for word in tokens if word not in custom_stopwords]  # Sử dụng danh sách tùy chỉnh
    return ' '.join(tokens)

# chuẩn hoá dữ liệu
X = [preprocess(item["text"]) for item in training_data]
y = [item["intent"] for item in training_data]

# Vector hóa văn bản (Bag of Words)
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.5, random_state=42)

# Huấn luyện mô hình Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Hàm để dự đoán ý định của câu hỏi
def predict_intent(text):
    processed_text = preprocess(text)
    vectorized_text = vectorizer.transform([processed_text])
    # Lấy xác suất dự đoán
    probabilities = model.predict_proba(vectorized_text)[0]
    max_prob = max(probabilities)
    # Xác định ngưỡng
    threshold = 0.05  # Bạn có thể điều chỉnh ngưỡng này
    if max_prob >= threshold:
        prediction = model.classes_[probabilities.argmax()]
    else:
        prediction = "khong_xac_dinh_cau_hoi"
    return prediction

# API cho phép người dùng nhập câu hỏi và nhận kết quả
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')  # Nhận tin nhắn từ người dùng
    intent = predict_intent(user_input)  # Dự đoán ý định
    response = {
        "intent": intent,
        "message": answer_data[intent]
    }
    return jsonify(response)
if __name__ == '__main__':
    app.run(debug=True,port=3000)