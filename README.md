# Ứng dụng Dự đoán Bệnh Tim

Một ứng dụng machine learning toàn diện để dự đoán nguy cơ mắc bệnh tim sử dụng giao diện người dùng thân thiện được xây dựng bằng Python Tkinter.

## 🚀 Tính năng

- **Tải dữ liệu**: Nhập bộ dữ liệu CSV để phân tích
- **Tiền xử lý dữ liệu**: Làm sạch và chuẩn bị dữ liệu cho machine learning
- **Huấn luyện mô hình**: Huấn luyện các mô hình machine learning khác nhau
- **Phân tích**: Trực quan hóa hiệu suất mô hình và tầm quan trọng của đặc trưng
- **Phân tích tương quan**: Khám phá mối quan hệ giữa các biến
- **Giao diện tương tác**: Giao diện tab dễ sử dụng

## 📋 Yêu cầu

- Python 3.12+
- Các gói yêu cầu được liệt kê trong `requirements.txt`

## 🛠️ Cài đặt

### Tùy chọn 1: Cài đặt cục bộ

1. Clone repository:
```bash
git clone <repository-url>
cd Nhom21-PTDLL
```

2. Tạo môi trường ảo:
```bash
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate
```

3. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

4. Chạy ứng dụng:
```bash
python main.py
```

### Tùy chọn 2: Cài đặt Docker

#### Yêu cầu tiên quyết
- Docker đã được cài đặt trên hệ thống
- Docker Compose (tùy chọn, để quản lý dễ dàng hơn)

#### Sử dụng Docker

1. Build Docker image:
```bash
docker build -t heart-disease-predictor .
```

2. Chạy container:
```bash
# Cho Linux/Mac với X11 forwarding
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd)/data:/app/data \
  heart-disease-predictor

# Cho Windows (yêu cầu X11 server như Xming hoặc VcXsrv)
docker run -it --rm \
  -e DISPLAY=host.docker.internal:0.0 \
  -v %cd%\data:/app/data \
  heart-disease-predictor
```

#### Sử dụng Docker Compose

1. Chạy với Docker Compose (bao gồm X11 virtual framebuffer):
```bash
docker-compose up --build
```

2. Truy cập GUI:
   - Ứng dụng sẽ chạy trong một màn hình ảo
   - Kết nối qua VNC viewer tới `localhost:5900` để xem GUI
   - Mật khẩu VNC mặc định: (kiểm tra tài liệu container xvfb)

## 📁 Cấu trúc dự án

```
Nhom21-PTDLL/
├── main.py                 # Điểm khởi đầu chính của ứng dụng
├── requirements.txt        # Dependencies Python
├── Dockerfile             # Cấu hình Docker
├── docker-compose.yml     # Thiết lập Docker Compose
├── .dockerignore          # File ignore Docker
├── README.md              # File này
├── core/                  # Chức năng cốt lõi
│   ├── data_processor.py  # Tiện ích xử lý dữ liệu
│   └── model_trainer.py   # Huấn luyện mô hình machine learning
├── gui/                   # Thành phần GUI
│   ├── __init__.py
│   ├── analysis_tab.py
│   ├── correlation_tab.py
│   ├── data_loading_tab.py
│   ├── model_training_tab.py
│   └── preprocessing_tab.py
└── data/                  # File dữ liệu
    └── heart_attack_prediction_dataset.csv
```

## 🖥️ Cách sử dụng

1. **Tab Tải dữ liệu**: 
   - Nhấn "Chọn file CSV" để tải bộ dữ liệu
   - Xem trước cấu trúc dữ liệu và thống kê cơ bản

2. **Tab Tiền xử lý**:
   - Làm sạch và chuẩn bị dữ liệu
   - Xử lý giá trị thiếu
   - Phân loại biến (số, phân loại, nhị phân)

3. **Tab Huấn luyện mô hình**:
   - Chọn và huấn luyện các mô hình machine learning
   - Xem các chỉ số hiệu suất
   - Tạo đường cong ROC và ma trận nhầm lẫn

4. **Tab Phân tích**:
   - Phân tích hệ số mô hình và tầm quan trọng đặc trưng
   - Hiểu được đặc trưng nào đóng góp nhiều nhất cho dự đoán

5. **Tab Tương quan**:
   - Trực quan hóa tương quan giữa các biến
   - Tạo biểu đồ cặp và ma trận tương quan

## 🐳 Ghi chú Docker

### Ứng dụng GUI trong Docker

Ứng dụng này sử dụng Tkinter cho GUI, cần xử lý đặc biệt trong Docker:

- **Linux/Mac**: Sử dụng X11 forwarding để hiển thị GUI trên host
- **Windows**: Yêu cầu X11 server (Xming, VcXsrv, v.v.)
- **Headless**: Sử dụng Xvfb (X Virtual Framebuffer) cho môi trường headless

### Volume Mounts

Thiết lập Docker bao gồm volume mounts cho:
- `./data:/app/data` - Chia sẻ file dữ liệu giữa host và container
- `./output:/app/output` - Lưu đầu ra mô hình và kết quả

### Biến môi trường

- `DISPLAY`: Đặt cho X11 forwarding
- Có thể thêm các biến môi trường bổ sung trong `docker-compose.yml`

## 🔧 Phát triển

### Thêm Dependencies

1. Thêm gói mới vào `requirements.txt`
2. Rebuild Docker image:
```bash
docker-compose down
docker-compose up --build
```

### Cấu trúc mã

- `main.py`: Điểm khởi đầu ứng dụng và cửa sổ chính
- `core/`: Logic nghiệp vụ cốt lõi và chức năng ML
- `gui/`: Thành phần GUI được tổ chức theo tab
- Mỗi tab là một module riêng để dễ bảo trì

## 📊 Định dạng dữ liệu được hỗ trợ

- File CSV với bộ dữ liệu dự đoán bệnh tim
- Các cột yêu cầu có thể khác nhau tùy thuộc vào bộ dữ liệu cụ thể
- Bộ dữ liệu mẫu đi kèm: `heart_attack_prediction_dataset.csv`

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch
3. Thực hiện thay đổi
4. Test với Docker
5. Gửi pull request

## 📝 Giấy phép

[Thêm thông tin giấy phép ở đây]

## 🆘 Khắc phục sự cố

### Vấn đề Docker

1. **GUI không hiển thị trên Windows**:
   - Cài đặt và chạy X11 server (Xming, VcXsrv)
   - Đặt biến môi trường DISPLAY đúng cách

2. **Lỗi quyền truy cập bị từ chối**:
   - Kiểm tra quyền file trong volume được mount
   - Đảm bảo Docker có quyền truy cập thư mục yêu cầu

3. **Container không khởi động được**:
   - Kiểm tra Docker logs: `docker-compose logs`
   - Xác minh tất cả dependencies trong `requirements.txt`

### Vấn đề ứng dụng

1. **Tải dữ liệu thất bại**:
   - Đảm bảo định dạng file CSV đúng
   - Kiểm tra quyền file

2. **Lỗi huấn luyện mô hình**:
   - Xác minh tiền xử lý dữ liệu đã hoàn thành
   - Kiểm tra giá trị thiếu hoặc kiểu dữ liệu không đúng

## 📞 Hỗ trợ

Cho các vấn đề và câu hỏi:
- Tạo issue trong repository
- Kiểm tra tài liệu hiện có
- Xem lại error logs trong ứng dụng

---

**Lưu ý**: Ứng dụng này được thiết kế cho mục đích giáo dục và nghiên cứu trong dự đoán bệnh tim sử dụng các kỹ thuật machine learning.
