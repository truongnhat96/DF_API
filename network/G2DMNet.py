import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from .GSFTM import GSFTM
from .CSMIAM import MultiScaleCSMIAM
from .DECAM import DECAM


class G2DMNet(nn.Module):
    def __init__(self,
                 num_classes=2,
                 image_size=(224, 224),
                 pretrain=False,
                 use_gsftm=True,
                 use_csmiam=True,
                 use_decam=True,
                 # Tính năng mới: Các thông số DECAM được hiển thị giúp việc điều chỉnh thông số dễ dàng hơn.
                 decam_Tf=0.5,
                 decam_Tp=0.3,
                 decam_beta_init=1.0,
                 decam_high_freq_weight=0.6,
                 # Tính năng mới: Tham số trọng số tần số cao của GSFTM (giải quyết lỗi hiện tại)
                 gsftm_high_freq_weight=0.8):  # Tham số mới
        super(G2DMNet, self).__init__()
        self.pretrain = pretrain
        self.use_gsftm = use_gsftm
        self.use_csmiam = use_csmiam
        self.use_decam = use_decam

        # 1. Mô-đun mô hình hóa đặc trưng chung tần số không gian Gabor (chuyển trọng số tần số cao)
        self.gsftm = GSFTM(
            image_size=image_size,
            num_scales=3,
            num_orientations=6,
            ksize=21,
            high_freq_weight=gsftm_high_freq_weight  # Chuyển tham số GSFTM
        ) if use_gsftm else None

        # 2. Lớp chuyển đổi đặc trưng đa tỉ lệ (điều chỉnh đầu vào dựa trên việc có sử dụng GSFTM hay không)
        in_channels = 128 if use_gsftm else 3
        self.scale_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=2, groups=64, dilation=2),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # 3. Mô-đun chú ý thông tin tương hỗ không gian kênh đa quy mô
        self.csmiam = MultiScaleCSMIAM(
            in_channels=64,
            kernel_sizes=[1, 3, 5]
        ) if use_csmiam else None

        # 4. Lớp tích chập để hợp nhất đặc trưng
        self.fusion_conv = nn.Conv2d(64, 128, kernel_size=1)

        # 5. Mô hình cơ sở (EfficientNet-B0)
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = self.base_model.features
        self.avgpool = self.base_model.avgpool

        # 6. Lớp chiếu kênh (phù hợp với đầu vào EfficientNet)
        self.channel_proj = nn.Conv2d(128, 3, 1)

        # 7. Mô-đun chú ý bất định hợp nhất phân lớp động (DECAM)
        self.decam = None
        if use_decam and not pretrain:
            self.decam = DECAM(
                T_f=decam_Tf,
                T_p=decam_Tp,
                beta_init=decam_beta_init,
                high_freq_weight=decam_high_freq_weight
            )

        # 8. Bộ tạo bản đồ xác suất hỗ trợ DECAM
        self.prob_head = None
        if use_decam:
            self.prob_head = nn.Sequential(
                nn.Conv2d(1280, 2, kernel_size=1),
                nn.Softmax(dim=1)
            )

        # 9. Lớp phân loại
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        original_x = x  # Lưu đầu vào gốc
        # Giai đoạn 1: Trích xuất đặc trưng không gian-tần số Gabor
        if self.use_gsftm and self.gsftm is not None:
            gsftm_feat = self.gsftm(x)
        else:
            gsftm_feat = original_x

        # Giai đoạn 2: Chuyển đổi đặc trưng đa tỉ lệ
        multi_scale_feat = self.scale_conv(gsftm_feat)

        # Giai đoạn 3: Chú ý thông tin tương hỗ không gian kênh đa quy mô
        if self.use_csmiam and self.csmiam is not None:
            att_feat = self.csmiam(multi_scale_feat)
        else:
            att_feat = multi_scale_feat

        # Giai đoạn 4: Hợp nhất đặc trưng
        fused_feat = self.fusion_conv(att_feat)

        # Giai đoạn 5: Trích xuất đặc trưng mô hình cơ sở
        base_input = self.channel_proj(fused_feat)
        base_feat = self.features(base_input)  # (B, 1280, 7, 7)

        # Giai đoạn 6: Tạo bản đồ xác suất dự đoán (chỉ khi DECAM được bật)
        prob_map = None
        if self.use_decam and self.prob_head is not None:
            prob_map = self.prob_head(base_feat)  # (B, 2, 7, 7)

        # Giai đoạn 7: DECAM chú ý trọng số (chỉ khi huấn luyện chính thức và được bật)
        decam_feat = base_feat  # Mặc định sử dụng đặc trưng cơ sở
        if (not self.pretrain) and self.use_decam and self.decam is not None:
            assert prob_map is not None, "Khi DECAM được bật, prob_map phải không rỗng"
            decam_feat, _ = self.decam(base_feat, prob_map)

        # Giai đoạn 8: Phân loại
        x = self.avgpool(decam_feat)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        return logits, decam_feat

    # Phương thức mới để trích xuất đặc trưng và bản đồ xác suất cho DECAM
    def extract_features(self, x):
        """
        Trích xuất đặc trưng và bản đồ xác suất từ mô hình
        Đầu vào: x - Tensor hình ảnh đầu vào, hình dạng (B, 3, H, W)
        Trả về: logits, features, prob_map
        """
        original_x = x  # Lưu đầu vào gốc

        # Giai đoạn 1: Trích xuất đặc trưng không gian-tần số Gabor
        if self.use_gsftm and self.gsftm is not None:
            gsftm_feat = self.gsftm(x)
        else:
            gsftm_feat = original_x

        # Giai đoạn 2: Chuyển đổi đặc trưng đa tỉ lệ
        multi_scale_feat = self.scale_conv(gsftm_feat)

        # Giai đoạn 3: Chú ý thông tin tương hỗ không gian kênh đa quy mô
        if self.use_csmiam and self.csmiam is not None:
            att_feat = self.csmiam(multi_scale_feat)
        else:
            att_feat = multi_scale_feat

        # Giai đoạn 4: Hợp nhất đặc trưng
        fused_feat = self.fusion_conv(att_feat)

        # Giai đoạn 5: Trích xuất đặc trưng mô hình cơ sở
        base_input = self.channel_proj(fused_feat)
        base_feat = self.features(base_input)  # (B, 1280, 7, 7)

        # Giai đoạn 6: Tạo bản đồ xác suất dự đoán (chỉ khi DECAM được bật)
        prob_map = None
        if self.use_decam and self.prob_head is not None:
            prob_map = self.prob_head(base_feat)  # (B, 2, 7, 7)

        # Giai đoạn 7: Đầu ra đầu phân loại (để giữ nhất quán giao diện)
        x = self.avgpool(base_feat)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        # Trả về kết quả phân loại, bản đồ đặc trưng, và bản đồ xác suất (để phù hợp với yêu cầu gọi calculate_tf_tp)
        return logits, base_feat, prob_map
    