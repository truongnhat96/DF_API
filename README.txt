# HƯỚNG DẪN CÀI ĐẶT VÀ KHỞI CHẠY API

1. Cài đặt Python (khuyến nghị 3.9 - 3.11)
2. Mở Terminal/CMD tại thư mục này và cài đặt thư viện:
   pip install -r requirements.txt

3. Chạy Server:
   - Dùng XceptionNet (Cấu hình mặc định hiện tại):
     uvicorn api:app --host 0.0.0.0 --port 8000

   - Dùng G2DMNet (Windows PowerShell):
     $env:MODEL_TYPE="G2DMNet"; uvicorn api:app --host 0.0.0.0 --port 8000

4. Cách Test:
   - Mở trình duyệt vào: http://localhost:8000/docs
   - Hoặc dùng Postman -> Gửi POST request tới http://localhost:8000/predict
     với body dạng form-data, chọn Key: `file` (dạng File) và tải ảnh lên.