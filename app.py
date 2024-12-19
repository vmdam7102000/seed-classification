import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Tải mô hình đã huấn luyện
MODEL_PATH = 'seed_classifier_model.h5'
model = load_model(MODEL_PATH)

# Cập nhật danh sách các loại hạt
class_indices = {
    'Almond': 0, 'Brazil Nut': 1, 'Cashew': 2, 'Chestnut': 3,
    'Hazelnut': 4, 'Macadamia': 5, 'Peanut': 6, 'Pecan': 7,
    'Pine Nut': 8, 'Pistachio': 9, 'Walnut': 10
}
# Ánh xạ chỉ số thành tên lớp
labels = {v: k for k, v in class_indices.items()}

# Thiết lập giao diện ứng dụng
st.set_page_config(page_title="Phân loại hạt", layout="centered")
st.title("🌰 Ứng dụng phân loại hạt")
st.write("Tải lên một hình ảnh của hạt để dự đoán loại hạt.")

# Bộ tải ảnh
uploaded_file = st.file_uploader("Chọn hình ảnh (định dạng .jpg, .jpeg, .png):", type=["jpg", "jpeg", "png"])

# Kích thước ảnh chuẩn
IMG_HEIGHT, IMG_WIDTH = 150, 150

# Hàm tiền xử lý ảnh
def preprocess_image(image):
    """Tiền xử lý ảnh: resize, chuyển đổi thành mảng và chuẩn hóa."""
    img = image.resize((IMG_WIDTH, IMG_HEIGHT))  # Thay đổi kích thước ảnh
    img_array = img_to_array(img)  # Chuyển đổi thành mảng
    img_array = img_array / 255.0  # Chuẩn hóa về khoảng [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch
    return img_array

# Nếu người dùng tải lên hình ảnh
if uploaded_file is not None:
    # Hiển thị hình ảnh được tải lên
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Hình ảnh đã tải lên", use_column_width=True)
    st.write("⏳ Đang phân loại...")

    # Tiền xử lý ảnh
    input_image = preprocess_image(image)

    # Dự đoán
    predictions = model.predict(input_image)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Lấy chỉ số lớp dự đoán cao nhất

    # Kiểm tra và lấy tên lớp tương ứng
    if predicted_class in labels:
        predicted_label = labels[predicted_class]
        # Hiển thị kết quả dự đoán
        st.success(f"✅ **Kết quả dự đoán:** {predicted_label}")
    else:
        st.error(f"❌ Lỗi: Không tìm thấy chỉ số lớp {predicted_class} trong ánh xạ lớp.")

    # Hiển thị điểm tin cậy
    st.write("📊 **Điểm tin cậy cho từng loại hạt:**")
    for idx, score in enumerate(predictions[0]):
        st.write(f"- {labels.get(idx, 'Không xác định')}: {score:.2%}")
else:
    # Hiển thị thông báo nếu chưa tải lên ảnh
    st.info("🖼️ Vui lòng tải lên một hình ảnh để bắt đầu.")
