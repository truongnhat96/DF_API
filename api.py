"""
DeepFake Detection – FastAPI Backend (Ensemble Pipeline)
=========================================================
Endpoint duy nhất: POST /predict
  - Nhận URL/File ảnh từ client mobile
  - Download ảnh, tiền xử lý (MTCNN face crop)
  - Chạy inference với 4 model ensemble:
      1. XceptionNet (local checkpoint)
      2. SigLIP2 Deepfake Detector (HuggingFace)
      3. CommunityForensics ViT (HuggingFace)
      4. CLIP-Yermakov Deepfake Detector (TorchScript từ HuggingFace)
  - Trả về điểm ensemble + từng model riêng lẻ

Thư viện cần thiết:
    pip install fastapi uvicorn[standard] httpx python-multipart
    pip install torch torchvision transformers timm
    pip install opencv-python facenet-pytorch numpy pillow
    pip install huggingface_hub
  
Khởi chạy:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Biến môi trường tùy chọn:
    MODEL_PATH  – đường dẫn tới file .pth XceptionNet (mặc định: xem DEFAULT_MODEL_PATH_XCEPTION)
    DEVICE      – "cuda" | "cpu" (mặc định: tự phát hiện)
"""

import io
import ipaddress
import os
import socket
import sys
import traceback
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import urlparse

import httpx
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, field_validator
from torchvision import transforms

# Đảm bảo project root trong sys.path để import network/XceptionNet
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network.XceptionNet import XceptionNet  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Cấu hình
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH_XCEPTION = os.path.join(
    "model",
    "Deepfakes_pretrain_multi",
    "xception_best.pth",
)

MODEL_PATH: str = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH_XCEPTION)
DEVICE: str = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE: tuple[int, int] = (256, 256)  # kích thước ảnh đầu vào cho XceptionNet
CLASS_NAMES: list[str] = ["Real", "Fake"]

# Thư mục lưu weights tải xuống từ HuggingFace (offline-friendly)
HF_LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "hf_weights")

# Danh sách Content-Type được chấp nhận khi tải ảnh
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"}

# Giới hạn kích thước file ảnh: 20 MB
MAX_IMAGE_BYTES = 20 * 1024 * 1024

# Lưu ảnh debug sau tiền xử lý (mặc định tắt để tránh trigger auto-reload của frontend dev server)
SAVE_DEBUG_FACE = os.environ.get("SAVE_DEBUG_FACE", "0") == "1"

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Tiền Xử Lý (Face Crop + Direct Resize)
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
      3. Crop vuông, 100% pixel trong output là pixel GỐC từ ảnh ban đầu.
      
    Trả về PIL.Image (RGB) đã crop – CHƯA resize, CHƯA normalize.
    Việc resize & normalize sẽ được thực hiện riêng cho từng model.
    """

    def __init__(self, device='cpu'):
        self.device = device
        # landmarks=True để MTCNN trả về tọa độ 5 điểm (mắt trái, mắt phải, mũi, 2 mép miệng)
        self.mtcnn = MTCNN(keep_all=False, select_largest=True, device=self.device)

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

    def extract_face(self, image_bytes: bytes) -> Image.Image:
        """Pipeline chính: bytes ảnh → PIL.Image (face crop vuông, chưa resize)."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Không thể decode định dạng ảnh.")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Crop face vuông — chỉ lấy khuôn mặt, không lấy nền
        cropped_face = self.detect_and_crop_square(image_rgb)

        h, w = cropped_face.shape[:2]
        if SAVE_DEBUG_FACE:
            debug_path = os.path.join(os.getcwd(), "debug_processed_face.jpg")
            cv2.imwrite(debug_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
            print(f"[Pipeline] Debug: {debug_path} | face crop {w}x{h}")
        else:
            print(f"[Pipeline] Face crop size: {w}x{h}")

        # Chuyển sang PIL Image (RGB)
        return Image.fromarray(cropped_face)


# Khởi tạo singleton pipeline
preprocess_pipeline = DeepfakePreprocessingPipeline(device=DEVICE)


# ──────────────────────────────────────────────────────────────────────────────
# Transforms riêng cho từng model
# ──────────────────────────────────────────────────────────────────────────────

def _make_transform(size: int) -> transforms.Compose:
    """Tạo transform chuẩn: Resize → ToTensor → Normalize(0.5, 0.5)."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

# XceptionNet: 256×256
transform_xception = _make_transform(256)

# SigLIP2: dùng AutoImageProcessor riêng (sẽ load khi model được load)
# CommunityForensics: 384×384
transform_cf = _make_transform(384)

# CLIP-Yermakov: dùng CLIPProcessor riêng (sẽ load khi model được load)


# ──────────────────────────────────────────────────────────────────────────────
# Model Registry – load tất cả 4 model một lần duy nhất khi server khởi động
# ──────────────────────────────────────────────────────────────────────────────
_models: dict = {}


def _load_xception() -> dict:
    """Load XceptionNet từ local checkpoint."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy file model XceptionNet: {MODEL_PATH}. "
            "Kiểm tra biến môi trường MODEL_PATH hoặc đường dẫn mặc định."
        )
    
    print(f"[Loader] Loading XceptionNet from {MODEL_PATH}...")
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
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Loader] XceptionNet loaded | Params: {param_count:,}")
    
    return {"model": model, "name": "XceptionNet"}


def _load_siglip2() -> dict:
    """
    Load SigLIP2 Deepfake Detector từ HuggingFace.
    
    Mapping theo model card:
      - Class 0: "Fake"
      - Class 1: "Real"
    → fake_index = 0
    """
    from transformers import SiglipForImageClassification, AutoImageProcessor
    
    repo_id = "prithivMLmods/Deepfake-Detect-Siglip2"
    cache_dir = os.path.join(HF_LOCAL_DIR, "siglip2")
    
    print(f"[Loader] Loading SigLIP2 from {repo_id}...")
    model = SiglipForImageClassification.from_pretrained(
        repo_id,
        cache_dir=cache_dir,
        local_files_only=True,
    ).to(DEVICE)
    model.eval()
    
    processor = AutoImageProcessor.from_pretrained(
        repo_id,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Loader] SigLIP2 loaded | Params: {param_count:,}")
    
    # Đọc id2label từ config để xác nhận
    id2label = getattr(model.config, "id2label", {})
    print(f"[Loader] SigLIP2 id2label: {id2label}")
    
    return {
        "model": model,
        "processor": processor,
        "name": "SigLIP2",
        # Theo model card: 0=Fake, 1=Real → fake_index = 0
        "fake_index": 0,
    }


def _load_community_forensics() -> dict:
    """
    Load CommunityForensics ViT từ HuggingFace.
    
    ⚠️ Bắt buộc trust_remote_code=True và input 384×384.
    """
    from transformers import AutoModelForImageClassification
    
    repo_id = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
    cache_dir = os.path.join(HF_LOCAL_DIR, "community_forensics")
    
    print(f"[Loader] Loading CommunityForensics ViT from {repo_id}...")
    model = AutoModelForImageClassification.from_pretrained(
        repo_id,
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=True,
    ).to(DEVICE)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Loader] CommunityForensics loaded | Params: {param_count:,}")
    
    # Đọc id2label nếu có
    id2label = getattr(model.config, "id2label", {})
    print(f"[Loader] CommunityForensics id2label: {id2label}")
    
    return {
        "model": model,
        "name": "CommunityForensics",
        # Giả thiết chuẩn: 0=Real, 1=Fake (sẽ xác nhận qua id2label)
        "fake_index": 1,
    }


def _load_clip_yermakov() -> dict:
    """
    Load CLIP-Yermakov Deepfake Detector (TorchScript) từ HuggingFace.
    
    Theo inference_torchscript.py chính thức:
      - softmax output: [p_real, p_fake]
      → fake_index = 1
    
    Sử dụng CLIPProcessor từ openai/clip-vit-large-patch14 để preprocessing
    (theo đúng hướng dẫn trong inference_torchscript.py).
    """
    from huggingface_hub import hf_hub_download
    from transformers import CLIPProcessor
    
    repo_id = "yermandy/deepfake-detection"
    filename = "model.torchscript"
    cache_dir = os.path.join(HF_LOCAL_DIR, "clip_yermakov")
    
    print(f"[Loader] Checking local cache for CLIP-Yermakov TorchScript from {repo_id}...")
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=cache_dir,
        local_files_only=True,
    )
    
    print(f"[Loader] Loading CLIP-Yermakov from {model_path}...")
    model = torch.jit.load(model_path, map_location=DEVICE)
    model.eval()
    
    # Load CLIPProcessor chính hãng cho preprocessing
    print(f"[Loader] Loading CLIPProcessor (openai/clip-vit-large-patch14)...")
    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14",
        cache_dir=cache_dir,
        local_files_only=True,
    )
    
    print(f"[Loader] CLIP-Yermakov loaded successfully")
    
    return {
        "model": model,
        "processor": clip_processor,
        "name": "CLIP-Yermakov",
        "fake_index": 0,
    }


def load_all_models():
    """Load tất cả 4 model. Mỗi model được load độc lập, nếu 1 model lỗi vẫn load các model còn lại."""
    global _models
    
    loaders = {
        "xception": _load_xception,
        "siglip": _load_siglip2,
        "cf": _load_community_forensics,
        "clip": _load_clip_yermakov,
    }
    
    for key, loader_fn in loaders.items():
        try:
            _models[key] = loader_fn()
        except Exception as exc:
            print(f"[Loader] WARNING: KHONG THE load model '{key}': {exc}")
            traceback.print_exc()
    
    print(f"")
    print(f"[Loader] ===========================================")
    print(f"[Loader] Total models loaded: {len(_models)}/{len(loaders)}")
    for key, info in _models.items():
        print(f"[Loader]   OK  {key}: {info['name']}")
    print(f"[Loader] Device: {DEVICE}")
    print(f"[Loader] ===========================================\n")


# ──────────────────────────────────────────────────────────────────────────────
# Inference functions – mỗi model có logic riêng
# ──────────────────────────────────────────────────────────────────────────────

def predict_xception(face_pil: Image.Image) -> Optional[float]:
    """Inference XceptionNet: 256×256, trả về xác suất FAKE (0-1)."""
    info = _models.get("xception")
    if info is None:
        return None
    
    model = info["model"]
    tensor = transform_xception(face_pil).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(tensor)
        # XceptionNet trả về (logits, feats)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        probs = F.softmax(logits, dim=1)[0]
    
    # XceptionNet: index 0 = Real, index 1 = Fake
    return float(probs[1].item())


def predict_siglip(face_pil: Image.Image) -> Optional[float]:
    """
    Inference SigLIP2: sử dụng AutoImageProcessor riêng.
    
    Theo model card:
      - Class 0 = "Fake"
      - Class 1 = "Real"
    → fake_index = 0
    """
    info = _models.get("siglip")
    if info is None:
        return None
    
    model = info["model"]
    processor = info["processor"]
    fake_index = info["fake_index"]
    
    # Sử dụng processor chính hãng từ model để đảm bảo preprocessing đúng
    inputs = processor(images=face_pil.convert("RGB"), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)
    
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        probs = F.softmax(outputs.logits, dim=1)[0]
    
    return float(probs[fake_index].item())


def predict_cf(face_pil: Image.Image) -> Optional[float]:
    """
    Inference CommunityForensics ViT: 384×384 (bắt buộc).
    """
    info = _models.get("cf")
    if info is None:
        return None
    
    model = info["model"]
    fake_index = info["fake_index"]
    
    tensor = transform_cf(face_pil.convert("RGB")).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(pixel_values=tensor)
        probs = F.softmax(outputs.logits, dim=1)[0]
    
    return float(probs[fake_index].item())


def predict_clip(face_pil: Image.Image) -> Optional[float]:
    """
    Inference CLIP-Yermakov (TorchScript): sử dụng CLIPProcessor riêng.
    
    Theo official inference_torchscript.py:
      - softmax output: [p_real, p_fake]
      → fake_index = 1
    """
    info = _models.get("clip")
    if info is None:
        return None
    
    model = info["model"]
    processor = info["processor"]
    fake_index = info["fake_index"]
    
    # Sử dụng CLIPProcessor chính hãng (openai/clip-vit-large-patch14)
    inputs = processor(images=face_pil.convert("RGB"), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)
    
    with torch.no_grad():
        output = model(pixel_values)
        probs = F.softmax(output, dim=1)[0]
    
    return float(probs[fake_index].item())


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble logic – Khối quyết định theo thứ tự ưu tiên
# ──────────────────────────────────────────────────────────────────────────────

def priority_based_decision(scores: dict[str, Optional[float]]) -> dict:
    """
    Hệ thống ra quyết định theo mức độ ưu tiên cụ thể trên từng model:
    Chỉ kết luận FAKE khi:
      - Xception có fake_score >= 0.65 ĐỒNG THỜI CF và CLIP > 0.75
      - HOẶC Xception > 0.90 ĐỒNG THỜI 1 trong 2 model (CF, CLIP) > 0.85
    Chỉ kết luận REAL khi:
      - Xception < 0.65 VÀ SigLIP dự đoán REAL
      - HOẶC (CF < 0.75 HOẶC CLIP < 0.75) VÀ SigLIP dự đoán REAL
    Ngoại lệ: So sánh fake_strength và real_confidence. Nếu bằng nhau, trả về UNCERTAIN và log forced output.
    """
    x = scores.get("xception")
    s = scores.get("siglip")
    c = scores.get("cf")
    l = scores.get("clip")
    
    # Tính fake_strength (đóng vai trò chỉ số hỗ trợ show response)
    weights = {'xception': 0.60, 'siglip': 0.20, 'cf': 0.10, 'clip': 0.10}
    strength_num = sum((scores[k] * weights[k]) for k in weights if scores.get(k) is not None)
    strength_den = sum(weights[k] for k in weights if scores.get(k) is not None)
    fake_strength = (strength_num / strength_den) if strength_den > 0 else 0.0

    r = (1.0 - s) if s is not None else 0.5
    real_confidence = r

    siglip_real = (s is not None and s < 0.50) # Tức là dự đoán REAL

    is_fake = False
    is_real = False
    reason = ""

    # ── Điều kiện FAKE ──
    # C1: x >= 0.65 AND c > 0.75 AND l > 0.75
    cond_fake_1 = (x is not None and x >= 0.65) and (c is not None and c > 0.75) and (l is not None and l > 0.75)
    # C2: x > 0.90 AND (c > 0.85 OR l > 0.85)
    cond_fake_2 = (x is not None and x > 0.90) and ((c is not None and c > 0.8) or (l is not None and l > 0.8))

    if cond_fake_1 or cond_fake_2:
        is_fake = True
        if cond_fake_1:
            reason = "FAKE: x >= 0.65 đồng thời c > 0.75 và l > 0.75"
        else:
            reason = "FAKE: x > 0.90 và (c > 0.85 hoặc l > 0.85)"

    # ── Điều kiện REAL ──
    # C1: x < 0.65 AND siglip_real
    cond_real_1 = (x is not None and x < 0.65) and siglip_real
    # C2: (c < 0.75 OR l < 0.75) AND siglip_real
    cond_real_2 = ((c is not None and c < 0.75) or (l is not None and l < 0.75)) and siglip_real

    if not is_fake and (cond_real_1 or cond_real_2):
        is_real = True
        if cond_real_1:
            reason = "REAL: x < 0.65 và siglip chọn REAL"
        else:
            reason = "REAL: (c < 0.75 hoặc l < 0.75) và siglip chọn REAL"
            
    # ── Quyết định ──
    if is_fake:
        label = "FAKE"
        if cond_fake_1:
            # Dựa vào x, c, l
            values = [v for v in [x, c, l] if v is not None]
            confidence = sum(values) / len(values) if values else fake_strength
        else:
            # cond_fake_2: x và (1 trong 2 CF, CLIP)
            values = [x]
            max_other = max([v for v in [c, l] if v is not None], default=None)
            if max_other is not None:
                values.append(max_other)
            confidence = sum(values) / len(values) if values else fake_strength
            
    elif is_real:
        label = "REAL"
        if cond_real_1:
            # Dựa vào x và siglip real (s)
            values = [(1.0 - v) for v in [x, s] if v is not None]
            confidence = sum(values) / len(values) if values else real_confidence
        else:
            # cond_real_2: siglip real (s) và (1 trong 2 CF, CLIP)
            values = [1.0 - s] if s is not None else []
            min_other = min([v for v in [c, l] if v is not None], default=None)
            if min_other is not None:
                values.append(1.0 - min_other)
            confidence = sum(values) / len(values) if values else real_confidence
            
    else:
        # Fallback
        if fake_strength > real_confidence:
            label = "FAKE"
            reason = f"Fallback FAKE: fake_strength ({fake_strength:.4f}) > real_confidence ({real_confidence:.4f})"
            confidence = fake_strength
        elif real_confidence > fake_strength:
            label = "REAL"
            reason = f"Fallback REAL: real_confidence ({real_confidence:.4f}) > fake_strength ({fake_strength:.4f})"
            confidence = real_confidence
        else:
            label = "UNCERTAIN"
            reason = (f"Tie break ({fake_strength:.4f} == {real_confidence:.4f}): "
                      f"x={x}, s={s}, c={c}, l={l}. "
                      f"-> Forced output: [REAL, FAKE] (Equal)")
            confidence = 0.0

    # Tính fake_votes hỗ trợ hiển thị
    THRESHOLD_XCEPTION = 0.65
    THRESHOLD_SIGLIP   = 0.50
    THRESHOLD_CF       = 0.60
    THRESHOLD_CLIP     = 0.60
    x_fake = (x >= THRESHOLD_XCEPTION) if x is not None else False
    s_fake = (s >= THRESHOLD_SIGLIP) if s is not None else False
    c_fake = (c >= THRESHOLD_CF) if c is not None else False
    l_fake = (l >= THRESHOLD_CLIP) if l is not None else False
    fake_votes = sum([x_fake, s_fake, c_fake, l_fake])

    return {
        "label": label,
        "fake_strength": round(fake_strength, 4),
        "real_confidence": round(real_confidence, 4),
        "fake_votes": fake_votes,
        "confidence": round(confidence, 4),
        "reason": reason,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan – load TẤT CẢ model ngay khi server khởi động (warm-up)
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()
    yield


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DeepFake Detection API (Ensemble)",
    description=(
        "API phát hiện deepfake sử dụng ensemble 4 model: "
        "XceptionNet, SigLIP2, CommunityForensics ViT, CLIP-Yermakov. "
        "Gửi URL/file ảnh khuôn mặt, nhận kết quả dự đoán Real/Fake."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Cho phep frontend goi API tu origin khac (file://, localhost khac port, v.v.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────────
class ModelScore(BaseModel):
    """Kết quả dự đoán của một model đơn lẻ."""
    name: str              # tên model
    fake_score: Optional[float] = None  # xác suất FAKE (0-1), None nếu model không khả dụng
    available: bool = True  # model có được load thành công không


class PredictResponse(BaseModel):
    # Kết quả quyết định Priority-based
    label: str              # "FAKE" | "REAL" | "UNCERTAIN"
    confidence: float       # độ tin cậy (0-1)
    fake_strength: float    # sức mạnh tín hiệu FAKE tổng hợp (weighted: 60% xception, 20% siglip, 10% cf, 10% clip)
    real_confidence: float  # SigLIP2 real confidence (1 - siglip_fake_score)
    fake_votes: int         # số fake detector vượt ngưỡng (0-4)
    reason: str             # giải thích lý do ra quyết định
    
    # Chi tiết từng model (fake_score: 0-1)
    scores: dict[str, Optional[float]]  # {"xception": 0.85, "siglip": 0.02, "cf": 0.78, "clip": 0.71}
    
    # Metadata
    models_used: int         # số model thực sự chạy inference
    models_total: int        # tổng số model đã load
    device: str              # "cuda" hoặc "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint /predict
# ──────────────────────────────────────────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Dự đoán ảnh có phải deepfake không (Ensemble 4 model)",
    description=(
        "Nhận `image_url` qua form-data hoặc file ảnh trực tiếp qua `file` form-data, "
        "chạy ensemble 4 model và trả về kết quả dự đoán chi tiết."
    ),
)
async def predict(
    image_url: str = Form(None),
    file: UploadFile = File(None)
) -> PredictResponse:
    
    if not image_url and not file:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp `image_url` hoặc upload `file`.")
    
    if not _models:
        raise HTTPException(status_code=503, detail="Không có model nào được load. Kiểm tra log server.")
        
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

    # ── 2. Face Detection & Crop ─────────────────────────────────────────────
    try:
        face_pil = preprocess_pipeline.extract_face(image_content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi tiền xử lý ảnh: {exc}")

    # ── 3. Ensemble Inference ────────────────────────────────────────────────
    scores: dict[str, Optional[float]] = {}
    
    predict_fns = {
        "xception": predict_xception,
        "siglip": predict_siglip,
        "cf": predict_cf,
        "clip": predict_clip,
    }
    
    for key, fn in predict_fns.items():
        try:
            score = fn(face_pil)
            scores[key] = round(score, 4) if score is not None else None
            if score is not None:
                model_name = _models.get(key, {}).get("name", key)
                print(f"[Inference] {model_name}: fake_score = {score:.4f}")
        except Exception as exc:
            print(f"[Inference] ⚠️ Lỗi inference model '{key}': {exc}")
            traceback.print_exc()
            scores[key] = None

    # ── 4. Quyết định Priority-based ─────────────────────────────────────────────────
    models_used = sum(1 for s in scores.values() if s is not None)
    
    if models_used == 0:
        raise HTTPException(
            status_code=500,
            detail="Tat ca model deu that bai trong qua trinh inference."
        )
    
    decision = priority_based_decision(scores)
    
    print(f"[Decision] Label: {decision['label']} | "
          f"Fake votes: {decision['fake_votes']}/4 | "
          f"Fake strength: {decision['fake_strength']:.4f} | "
          f"SigLIP2 real: {decision['real_confidence']:.4f} | "
          f"Confidence: {decision['confidence']:.4f} | "
          f"Reason: {decision['reason']} | "
          f"Models: {models_used}/{len(_models)}")

    # ── 5. Trả về kết quả ────────────────────────────────────────────────────
    return PredictResponse(
        label=decision["label"],
        confidence=decision["confidence"],
        fake_strength=decision["fake_strength"],
        real_confidence=decision["real_confidence"],
        fake_votes=decision["fake_votes"],
        reason=decision["reason"],
        scores=scores,
        models_used=models_used,
        models_total=len(_models),
        device=DEVICE,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health", summary="Kiểm tra trạng thái server")
async def health():
    loaded_models = {k: v["name"] for k, v in _models.items()}
    return {
        "status": "ok",
        "models_loaded": len(_models),
        "models": loaded_models,
        "device": DEVICE,
    }
