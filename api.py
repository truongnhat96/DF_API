"""
DeepFake Detection – FastAPI Backend
=====================================
Endpoint duy nhất: POST /predict
  - Nhận URL/File ảnh từ client mobile
  - Download ảnh, tiền xử lý, chạy inference với G2DMNet/XceptionNet
  - Trả về toàn bộ kết quả dự đoán

Thư viện cần thiết:
    pip install fastapi uvicorn[standard] httpx
    pip install python-multipart  # nếu muốn hỗ trợ upload file ảnh trực tiếp (multipart/form-data)
    pip install opencv-python facenet-pytorch numpy
  
Khởi chạy:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Biến môi trường tùy chọn:
    MODEL_PATH  – đường dẫn tới file .pth (mặc định: xem DEFAULT_MODEL_PATH)
    DEVICE      – "cuda" | "cpu" (mặc định: tự phát hiện)
"""

import io
import ipaddress
import os
import socket
import sys
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import httpx
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from PIL import Image
from pydantic import BaseModel, field_validator
from torchvision import transforms

# Đảm bảo project root trong sys.path để import network/G2DMNet
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network.G2DMNet import G2DMNet  # noqa: E402
from network.XceptionNet import XceptionNet  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Cấu hình
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_TYPE = "XceptionNet" # Hoặc "XceptionNet" | "G2DMNet" (mặc định: XceptionNet nhẹ hơn, phù hợp cho mobile)
MODEL_TYPE: str = os.environ.get("MODEL_TYPE", DEFAULT_MODEL_TYPE)

DEFAULT_MODEL_PATH_G2DM = os.path.join(
    "model",
    "Deepfakes_pretrain_multi",
    # "pretrained_DeepFakes_FaceShifter_gsftmFalse_csmiamFalse_decamFalse_ratio0.7.pth",
    "best_Deepfakes1_gsftmTrue_csmiamTrue_decamTrue.pth",
)
DEFAULT_MODEL_PATH_XCEPTION = os.path.join(
    "model",
    "Deepfakes_pretrain_multi",
    "xception_best.pth",
)

if MODEL_TYPE == "XceptionNet":
    MODEL_PATH: str = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH_XCEPTION)
else:
    MODEL_PATH: str = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH_G2DM)

DEVICE: str = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE: tuple[int, int] = (256, 256)  # kích thước ảnh đầu vào cho model (phù hợp với pipeline huấn luyện)
CLASS_NAMES: list[str] = ["Real", "Fake"]

# Danh sách Content-Type được chấp nhận khi tải ảnh
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"}

# Giới hạn kích thước file ảnh: 20 MB
MAX_IMAGE_BYTES = 20 * 1024 * 1024

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Tiền Xử Lý (Face Crop + Letterbox Resize)
# ──────────────────────────────────────────────────────────────────────────────
import cv2
import numpy as np
from facenet_pytorch import MTCNN

class DeepfakePreprocessingPipeline:
    """
    Pipeline tiền xử lý ảnh cho DeepFake Detection.
    
    Chiến lược crop chính xác:
      1. MTCNN detect face → lấy bounding box + 5 facial landmarks.
      2. Dùng landmarks (2 mắt, mũi) để tính tâm khuôn mặt và khoảng cách mắt-mắt.
         Từ đó xác định vùng crop ĐÚNG khuôn mặt (trán → cằm, má trái → má phải).
      3. Crop vuông, resize thẳng 256×256, KHÔNG pad, KHÔNG viền.
      4. 100% pixel trong output là pixel GỐC từ ảnh ban đầu.
    """

    def __init__(self, target_size=(256, 256), device='cpu'):
        self.target_size = target_size
        self.device = device
        # landmarks=True để MTCNN trả về tọa độ 5 điểm (mắt trái, mắt phải, mũi, 2 mép miệng)
        self.mtcnn = MTCNN(keep_all=False, select_largest=True, device=self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def detect_and_crop_square(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Detect face bằng MTCNN, dùng bounding box gốc (đã khít mặt) để crop.
        
        MTCNN bounding box đã bao trọn khuôn mặt từ trán→cằm, má→má.
        Ta KHÔNG thêm margin ngoài nữa — chỉ crop đúng box rồi ép vuông
        bằng cách THU HẸP cạnh dài (cắt bớt) thay vì mở rộng cạnh ngắn (thêm nền).
        """
        img_h, img_w = image_rgb.shape[:2]
        boxes, probs, landmarks = self.mtcnn.detect(image_rgb, landmarks=True)

        if boxes is None:
            raise ValueError("Không tìm thấy khuôn mặt trong ảnh!")

        x1, y1, x2, y2 = boxes[0]
        box_w = x2 - x1
        box_h = y2 - y1
        
        # Tâm bounding box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Dịch tâm xuống dưới 5% chiều cao box để lấy thêm phần cằm
        # (vì ép vuông bằng min cắt đều 2 đầu → mất cằm nhiều hơn trán)
        cy += box_h * 0.05

        # Ép vuông bằng cạnh NGẮN nhất (= thu hẹp cạnh dài, cắt bớt vùng thừa)
        # Đây là điểm mấu chốt: dùng min thay vì max → không bao giờ lấy thêm nền
        side = min(box_w, box_h)
        half = side / 2

        # Tọa độ crop vuông, đặt tại tâm bounding box
        crop_x1 = int(cx - half)
        crop_y1 = int(cy - half)
        crop_x2 = int(cx + half)
        crop_y2 = int(cy + half)
        crop_side = crop_x2 - crop_x1

        # Shift khung vào trong ảnh nếu bị tràn biên
        if crop_x1 < 0:
            crop_x2 = min(crop_side, img_w)
            crop_x1 = 0
        if crop_y1 < 0:
            crop_y2 = min(crop_side, img_h)
            crop_y1 = 0
        if crop_x2 > img_w:
            crop_x1 = max(0, img_w - crop_side)
            crop_x2 = img_w
        if crop_y2 > img_h:
            crop_y1 = max(0, img_h - crop_side)
            crop_y2 = img_h

        return image_rgb[crop_y1:crop_y2, crop_x1:crop_x2]

    def process(self, image_bytes: bytes) -> torch.Tensor:
        """Pipeline chính: bytes ảnh → tensor 256×256 sẵn sàng inference."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Không thể decode định dạng ảnh.")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 1. Crop face vuông — chỉ lấy khuôn mặt, không lấy nền
        cropped_face = self.detect_and_crop_square(image_rgb)

        # 2. Resize thẳng về 256×256 (ảnh đã vuông → không méo)
        h, w = cropped_face.shape[:2]
        target_w, target_h = self.target_size
        if h > target_h or w > target_w:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC
        processed_img = cv2.resize(cropped_face, (target_w, target_h), interpolation=interpolation)

        # 3. Lưu debug
        debug_path = os.path.join(os.getcwd(), "debug_processed_face.jpg")
        cv2.imwrite(debug_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(f"[API] Debug: {debug_path} | crop {w}x{h} → {target_w}x{target_h}")

        # 4. Normalize → Tensor [1, 3, 256, 256]
        tensor = self.transform(processed_img).unsqueeze(0)
        return tensor

# Khởi tạo singleton pipeline
preprocess_pipeline = DeepfakePreprocessingPipeline(target_size=IMAGE_SIZE, device=DEVICE)

# ──────────────────────────────────────────────────────────────────────────────
# Model singleton (load một lần duy nhất khi server khởi động)
# ──────────────────────────────────────────────────────────────────────────────
_model = None


def _infer_flags_from_path(path: str) -> dict:
    """Tự động phát hiện use_gsftm / use_csmiam / use_decam từ tên file checkpoint.

    Quy ước tên file:
      *gsftmTrue*  hoặc *gsftmFalse*
      *csmiamTrue* hoặc *csmiamFalse*
      *decamTrue*  hoặc *decamFalse*
    Nếu không tìm thấy pattern → mặc định True (khớp với checkpoint v2).
    """
    basename = os.path.basename(path).lower()

    def _flag(keyword_true: str, keyword_false: str) -> bool:
        if keyword_false in basename:
            return False
        if keyword_true in basename:
            return True
        return True  # mặc định

    return {
        "use_gsftm":  _flag("gsftmtrue",  "gsftmfalse"),
        "use_csmiam": _flag("csmiamtrue", "csmiamfalse"),
        "use_decam":  _flag("decamtrue",  "decamfalse"),
    }


def get_model():
    """Trả về model đã được load (load lazy nếu chưa có)."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Không tìm thấy file model: {MODEL_PATH}. "
                "Kiểm tra biến môi trường MODEL_PATH hoặc đường dẫn mặc định."
            )
        
        if MODEL_TYPE == "XceptionNet":
            print(f"[API] Initializing XceptionNet with model path: {MODEL_PATH}")
            model = XceptionNet(num_classes=2).to(DEVICE)
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            # Remove "module." or "backbone." prefix if saved from DataParallel or custom wrapper
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k
                if new_key.startswith("module."):
                    new_key = new_key[7:]
                if new_key.startswith("backbone."):
                    new_key = new_key[9:]
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict)
        else:
            print(f"[API] Initializing G2DMNet with model path: {MODEL_PATH}")
            flags = _infer_flags_from_path(MODEL_PATH)
            print(f"[API] Checkpoint flags: {flags}")
            model = G2DMNet(
                num_classes=2,
                pretrain=False,
                **flags,
            ).to(DEVICE)
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict)
            
        model.eval()
        _model = model
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[API] Model loaded: {os.path.basename(MODEL_PATH)} | "
              f"Type: {MODEL_TYPE} | Device: {DEVICE} | Params: {param_count:,}")
    return _model


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan – load model ngay khi server khởi động (warm-up)
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_model()
    except FileNotFoundError as exc:
        print(f"[API] CẢNH BÁO: {exc} – model sẽ được load khi có request đầu tiên.")
    yield


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DeepFake Detection API",
    description=(
        "API phát hiện deepfake sử dụng mô hình G2DMNet và XceptionNet. "
        "Gửi URL ảnh khuôn mặt, nhận kết quả dự đoán Real/Fake."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    image_url: str

    @field_validator("image_url")
    @classmethod
    def validate_image_url(cls, v: str) -> str:
        """Chỉ chấp nhận URL http/https hợp lệ; chặn SSRF trỏ vào mạng nội bộ."""
        parsed = urlparse(v)

        # Kiểm tra scheme
        if parsed.scheme not in ("http", "https"):
            raise ValueError("image_url phải bắt đầu bằng http:// hoặc https://")

        # Kiểm tra host tồn tại
        hostname = parsed.hostname
        if not hostname:
            raise ValueError("image_url không hợp lệ: thiếu hostname")

        # Bảo vệ SSRF: giải địa chỉ IP của host rồi kiểm tra phạm vi private/loopback
        try:
            ip_str = socket.gethostbyname(hostname)
            ip = ipaddress.ip_address(ip_str)
        except socket.gaierror:
            raise ValueError(f"Không thể phân giải hostname: {hostname}")

        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("image_url không được trỏ tới địa chỉ IP nội bộ hoặc loopback")

        return v


class PredictResponse(BaseModel):
    prediction: str       # "Real" hoặc "Fake"
    label: int            # 0 = Real, 1 = Fake
    confidence: float     # xác suất của class được dự đoán (0–1)
    prob_real: float      # xác suất ảnh là thật
    prob_fake: float      # xác suất ảnh là deepfake
    model_type: str       # G2DMNet hoặc XceptionNet
    model_file: str       # tên file model đã dùng
    device: str           # "cuda" hoặc "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint /predict
# ──────────────────────────────────────────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Dự đoán ảnh có phải deepfake không",
    description=(
        "Nhận `image_url` qua JSON hoặc file ảnh trực tiếp qua `file` form-data, "
        "chạy model và trả về kết quả dự đoán chi tiết."
    ),
)
async def predict(
    image_url: str = Form(None),
    file: UploadFile = File(None)
) -> PredictResponse:
    
    if not image_url and not file:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp `image_url` hoặc upload `file`.")
        
    # ── 1. Đọc ảnh từ URL hoặc File ──────────────────────────────────────────
    image_content = b""
    
    if file:
        # Nhận ảnh từ file upload (multipart/form-data)
        if hasattr(file, 'content_type') and file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Content-Type không được hỗ trợ: '{file.content_type}'. "
                       f"Chấp nhận: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}",
            )
        
        image_content = await file.read()
        
        if len(image_content) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Ảnh quá lớn (tối đa {MAX_IMAGE_BYTES // (1024*1024)} MB).",
            )
            
    elif image_url:
        # Nhận ảnh từ URL
        try:
            # Validate URL cơ bản
            parsed = urlparse(image_url)
            if parsed.scheme not in ("http", "https"):
                raise ValueError("image_url phải bắt đầu bằng http:// hoặc https://")
            
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                response = await client.get(image_url)
            response.raise_for_status()
            
            # Kiểm tra Content-Type HTTP
            content_type = response.headers.get("content-type", "").split(";")[0].strip()
            if content_type not in ALLOWED_CONTENT_TYPES:
                raise HTTPException(
                    status_code=415,
                    detail=f"Content-Type từ URL không được hỗ trợ: '{content_type}'.",
                )
                
            image_content = response.content
            
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Tải ảnh thất bại (HTTP {exc.response.status_code}): {image_url}",
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Lỗi tải ảnh từ URL: {exc}")
            
        if len(image_content) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Ảnh tải về quá lớn (tối đa {MAX_IMAGE_BYTES // (1024*1024)} MB).",
            )

    # ── 2. Decode & tiền xử lý ───────────────────────────────────────────────
    try:
        tensor = preprocess_pipeline.process(image_content).to(DEVICE)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi tiền xử lý ảnh: {exc}")

    # ── 3. Inference ─────────────────────────────────────────────────────────
    try:
        model = get_model()
        with torch.no_grad():
            output = model(tensor)
            
            # G2DMNet trả về (logits, aux), XceptionNet trả về (logits, feats)
            # Trong cả hai trường hợp, logits là phần tử đầu tiên của tuple
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
                
            probs = F.softmax(logits, dim=1)[0]  # (2,)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi inference: {exc}")

    prob_real = float(probs[0])
    prob_fake = float(probs[1])
    label = int(probs.argmax().item())

    # ── 4. Trả về kết quả ────────────────────────────────────────────────────
    return PredictResponse(
        prediction=CLASS_NAMES[label],
        label=label,
        confidence=round(float(probs[label]), 4),
        prob_real=round(prob_real, 4),
        prob_fake=round(prob_fake, 4),
        model_type=MODEL_TYPE,
        model_file=os.path.basename(MODEL_PATH),
        device=DEVICE,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health", summary="Kiểm tra trạng thái server")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_type": MODEL_TYPE,
        "model_file": os.path.basename(MODEL_PATH),
        "device": DEVICE,
    }
