# Phát hiện bất thường trong ảnh y khoa
## Sinh viên thực hiện: Lê Thanh Truyền
## Lớp: DA20TTB
## MSSV: 110120163

## Mục tiêu
- Sử dụng các mô hình học sâu để tự động phát hiện bất thường trên ảnh y khoa.
- Đánh giá hiệu quả của các mô hình trong việc hỗ trợ chẩn đoán y khoa.

## Kiến trúc
- **Dữ liệu**: Sử dụng bộ dữ liệu ảnh X-quang để huấn luyện và kiểm tra mô hình.
- **Mô hình**: Triển khai ba mô hình chính: CNN, ResNet và DenseNet.
- **Giao diện**: Sử dụng Flask để xây dựng giao diện web cho phép người dùng tải lên ảnh và nhận kết quả dự đoán từ mô hình.

## Phần mềm cần thiết
- Python 3.9 trở lên
- Các thư viện Python: 
  - TensorFlow
  - Keras
  - Flask
  - Numpy
  - Matplotlib
  - Pandas  
  v.v...

## Cách thức triển khai và chạy chương trình

1. Cài đặt Python (Python 3.9 trở lên)

2. Clone repository từ GitHub:
    ```bash
    https://github.com/truyenda20ttb/tn-da20ttb-110120163-lethanhtruyen-phathienbatthuonganhykhoa.git
    cd tn-da20ttb-110120163-lethanhtruyen-phathienbatthuonganhykhoa/
    cd src
    cd GiaoDien  
    ```
3. Kích hoạt môi trường ảo
    ```bash
    .\.venv\Scripts\activate
    ```

4. Chạy ứng dụng Flask:
    ```bash
    python app.py
    ```

4. Truy cập vào địa chỉ `http://127.0.0.1:5000` để xem ứng dụng.

## Kiến trúc chi tiết

### Mô hình CNN
- Mô hình CNN được xây dựng với các lớp chập (convolutional layers) và lớp gộp (pooling layers) để trích xuất đặc trưng từ ảnh.

### Mô hình ResNet
- Mô hình ResNet sử dụng các khối còn dư (residual blocks) để cải thiện độ sâu của mô hình mà không gặp phải vấn đề gradient biến mất. Khoá luận sử dụng ResNet50

### Mô hình DenseNet
- Mô hình DenseNet kết nối mỗi lớp với tất cả các lớp trước đó, giúp cải thiện khả năng tái sử dụng đặc trưng và tăng độ chính xác. Khoá luận sử dụng DenseNet121

## Kết quả
- DenseNet: Độ chính xác 97.58%
- CNN: Độ chính xác 94.07%
- ResNet: Độ chính xác 95.45%

