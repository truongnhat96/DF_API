import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
# import matplotlib.pyplot as plt


class GSFTM(nn.Module):
    """（Gabor Spatial-Frequency Transformation Module）"""

    def __init__(
            self,
            image_size=(224, 224),
            num_scales=5,
            num_orientations=8,
            ksize=31,
            gamma=0.5,
            psi=0,
            device=None,
            min_sigma=0.1,
            # Tính năng mới: Tham số trọng số tần số cao để điều chỉnh ảnh hưởng của tần số cao
            high_freq_weight=0.8  # Tham số mới, mặc định 0.8
    ):
        super(GSFTM, self).__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size  # (H, W)
        self.num_scales = num_scales  # Số lượng tỉ lệ s
        self.num_orientations = num_orientations  # Số lượng hướng θ
        self.ksize = ksize  # Kích thước kernel Gabor không gian
        self.gamma = gamma  # Tỷ lệ elip
        self.psi = psi  # Độ lệch pha
        self.min_sigma = min_sigma  # Ngăn ngừa sigma quá nhỏ
        self.eps = 1e-10  # Tăng độ ổn định số học
        self.high_freq_weight = high_freq_weight  # Lưu tham số trọng số tần số cao

        # Tính tổng số bộ lọc
        self.total_filters = num_scales * num_orientations

        # 1. Khởi tạo bộ lọc Gabor không gian
        self.spatial_gabor_filters = self._create_spatial_gabor_filters()
        # Không gian đặc trưng hợp nhất 1x1 convolution (num_scales*num_orientations -> C_s)
        self.spatial_conv = nn.Conv2d(
            in_channels=self.total_filters,  # Số kênh đầu vào là tổng số bộ lọc
            out_channels=64,  # C_s có thể điều chỉnh theo nhu cầu
            kernel_size=1
        ).to(self.device)

        # 2. Khởi tạo bộ lọc Gabor tần số
        self.freq_gabor_filters = self._create_freq_gabor_filters()
        # Tần số đặc trưng hợp nhất chiếu (2*num_scales*num_orientations -> C_f)
        self.freq_proj = nn.Conv2d(
            in_channels=2 * self.total_filters,
            out_channels=64,  # C_f và C_s giữ nhất quán để hợp nhất
            kernel_size=1
        ).to(self.device)

        # 3. Các thành phần để phát hiện cạnh Sobel cho chỉ dẫn chéo miền
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).to(self.device)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).to(self.device)
        self._init_sobel_kernels()  # Khởi tạo bộ lọc Sobel

        # 4. Các phép tích chập hợp nhất 1x1 (C_s + C_f -> C_fusion)
        self.fusion_conv = nn.Conv2d(
            in_channels=64 + 64,
            out_channels=128,  # Số kênh đặc trưng hợp nhất
            kernel_size=1
        ).to(self.device)

    def _create_spatial_gabor_filters(self):
        """Tạo bộ lọc Gabor không gian"""
        filters = []
        for s in range(self.num_scales):
            # Tham số liên quan đến tỉ lệ: bước sóng λ tăng theo tỉ lệ
            lambd = torch.tensor(4 + 2 * s, device=self.device)
            sigma = lambd / 3.14  # Độ lệch chuẩn Gaussian liên quan đến bước sóng

            # Tính bước để không bao gồm điểm cuối np.pi
            step = np.pi / self.num_orientations
            thetas = torch.arange(0, np.pi, step, device=self.device)

            for theta in thetas:
                # Tạo một kernel Gabor không gian
                kernel = self._spatial_gabor_kernel(ksize=self.ksize, sigma=sigma, theta=theta, lambd=lambd)
                filters.append(kernel.unsqueeze(0).unsqueeze(0))  # Thêm chiều kênh (1,1,H,W)
        return torch.cat(filters, dim=0).to(self.device)  # Kết quả đầu ra: (S*θ, 1, H, W)

    def _spatial_gabor_kernel(self, ksize, sigma, theta, lambd):
        """Tạo một kernel Gabor không gian"""
        # Tạo lưới tọa độ
        x = torch.arange(ksize, device=self.device) - ksize // 2
        y = torch.arange(ksize, device=self.device) - ksize // 2
        x, y = torch.meshgrid(x, y, indexing='ij')

        # Xoay tọa độ
        x_theta = x * torch.cos(theta) + y * torch.sin(theta)
        y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

        # Công thức kernel Gabor
        gabor = torch.exp(-(x_theta **2 + self.gamma** 2 * y_theta **2) / (2 * sigma** 2)) \
                * torch.cos(2 * math.pi * x_theta / lambd + self.psi)

        # Chuẩn hóa
        gabor = gabor - gabor.mean()
        gabor = gabor / torch.norm(gabor)
        return gabor

    def _create_freq_gabor_filters(self):
        """Tạo bộ lọc Gabor tần số (thêm các biện pháp ổn định số)"""
        H, W = self.image_size
        u = torch.fft.fftfreq(W, device=self.device)
        v = torch.fft.fftfreq(H, device=self.device)
        U, V = torch.meshgrid(u, v, indexing='xy')

        filters = []
        for s in range(self.num_scales):
            lambd = torch.tensor(4 + 2 * s, device=self.device)
            # Giới hạn giá trị sigma tối thiểu để tránh không ổn định số
            sigma = max(lambd / 3.14, self.min_sigma)

            step = np.pi / self.num_orientations
            thetas = torch.arange(0, np.pi, step, device=self.device)

            for theta in thetas:
                U_rot = U * torch.cos(theta) + V * torch.sin(theta)
                V_rot = -U * torch.sin(theta) + V * torch.cos(theta)

                # Thêm giới hạn an toàn để tránh bùng nổ số mũ
                exponent = -0.5 * ((U_rot / (sigma / 2 + self.eps)) **2 + (V_rot / (sigma / 2 + self.eps))** 2)
                exponent = torch.clamp(exponent, min=-50, max=50)  # Ngăn chặn tràn exp

                envelope = torch.exp(exponent)
                modulation = torch.exp(1j * 2 * torch.pi * (1.0 / (lambd + self.eps)) * U_rot)

                freq_kernel = envelope * modulation

                # Chuẩn hóa an toàn
                norm = torch.norm(freq_kernel) + self.eps
                freq_kernel = freq_kernel / norm

                filters.append(freq_kernel.unsqueeze(0).unsqueeze(0))

        return torch.cat(filters, dim=0).to(self.device)

    def _init_sobel_kernels(self):
        """Khởi tạo bộ lọc phát hiện cạnh Sobel"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=self.device, dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=self.device, dtype=torch.float32)
        self.sobel_x.weight = nn.Parameter(sobel_x.unsqueeze(0).unsqueeze(0))
        self.sobel_y.weight = nn.Parameter(sobel_y.unsqueeze(0).unsqueeze(0))

    def _extract_spatial_features(self, x):
        """Trích xuất đặc trưng không gian: tích chập Gabor + hợp nhất"""
        B, C, H, W = x.shape

        # Áp dụng bộ lọc Gabor cho mỗi kênh, sau đó hợp nhất kết quả
        channel_features = []
        for c in range(C):
            # Ảnh đơn kênh: (B,1,H,W)
            x_c = x[:, c:c + 1, :, :]
            # Tích chập Gabor: (B, S*θ, H, W)
            conv_feat = F.conv2d(x_c, self.spatial_gabor_filters, padding=self.ksize // 2)
            channel_features.append(conv_feat)

        # Hợp nhất đặc trưng của tất cả các kênh bằng cách lấy trung bình theo chiều kênh
        spatial_feat = torch.stack(channel_features, dim=1).mean(dim=1)  # (B, total_filters, H, W)

        # Áp dụng tích chập 1x1 để hợp nhất đặc trưng
        spatial_feat = self.spatial_conv(spatial_feat)  # (B, 64, H, W)
        return spatial_feat

    def _extract_freq_features(self, x):
        """Trích xuất đặc trưng tần số: thêm các biện pháp ổn định số"""
        B, C, H, W = x.shape
        freq_feats = []
        for c in range(C):
            x_c = x[:, c:c + 1, :, :]
            x_fft = torch.fft.fftshift(torch.fft.fft2(x_c))

            scale_orient_feats = []
            for idx in range(self.total_filters):
                H_sθ = self.freq_gabor_filters[idx:idx + 1, :, :, :]
                filtered_fft = x_fft * H_sθ

                # Tính toán phổ biên độ an toàn hơn
                amp = torch.abs(filtered_fft)
                amp = torch.log1p(amp)  # log(1 + |z|) tránh log(0)

                phase = torch.angle(filtered_fft)
                scale_orient_feats.append(amp)
                scale_orient_feats.append(phase)

            freq_feat_c = torch.cat(scale_orient_feats, dim=1)
            freq_feats.append(freq_feat_c)

        freq_feat = torch.stack(freq_feats, dim=1).mean(dim=1)
        freq_feat = self.freq_proj(freq_feat)
        return freq_feat

    def _cross_domain_guidance(self, spatial_feat, freq_feat, amp_spectra):
        """Cơ chế hướng dẫn chéo miền: sử dụng tham số trọng số tần số cao"""
        B, C_s, H, W = spatial_feat.shape
        B, C_f, _, _ = freq_feat.shape

        # 1. Hướng dẫn không gian → tần số: phát hiện cạnh
        spatial_gray = spatial_feat.mean(dim=1, keepdim=True)
        edge_x = self.sobel_x(spatial_gray)
        edge_y = self.sobel_y(spatial_gray)

        # Thêm giới hạn an toàn để tránh bùng nổ gradient
        edge_x = torch.clamp(edge_x, -10, 10)
        edge_y = torch.clamp(edge_y, -10, 10)

        edge_map = torch.sqrt(edge_x **2 + edge_y** 2 + self.eps)  # Biên độ cạnh
        edge_map = F.interpolate(edge_map, size=(H, W), mode='bilinear', align_corners=True)
        edge_map = edge_map / (edge_map.max() + self.eps)  # Chuẩn hóa

        E_spatial = edge_map.repeat(1, C_f, 1, 1)
        freq_feat_guided = freq_feat * E_spatial

        # 2. Hướng dẫn tần số → không gian: năng lượng tần số cao (sử dụng trọng số tần số cao được cấu hình)
        h_cut = H // 4
        w_cut = W // 4

        # Tính năng lượng vùng tần số cao (chỉ giữ phần tần số cao)
        if amp_spectra[..., h_cut:-h_cut, w_cut:-w_cut].numel() > 0:
            # Áp dụng trọng số tần số cao: tăng ảnh hưởng của năng lượng tần số cao
            high_freq_amp = self.high_freq_weight * amp_spectra[..., h_cut:-h_cut, w_cut:-w_cut].mean(dim=[1, 2, 3], keepdim=True)
        else:
            high_freq_amp = torch.zeros_like(amp_spectra[:, :1, :1, :1])

        M_freq = F.interpolate(high_freq_amp, size=(H, W), mode='bilinear', align_corners=True)
        M_freq = M_freq.repeat(1, C_s, 1, 1)
        spatial_feat_guided = spatial_feat * M_freq

        return spatial_feat_guided, freq_feat_guided

    def forward(self, x):
        """GSFTM truyền tiến: đặc trưng không gian → đặc trưng tần số → hướng dẫn chéo miền → hợp nhất"""
        # 1. Trích xuất đặc trưng không gian
        spatial_feat = self._extract_spatial_features(x)  # (B, C_s, H, W)

        # 2. Trích xuất đặc trưng tần số (đồng thời giữ phổ biên độ để tạo mặt nạ tần số cao)
        B, C, H, W = x.shape
        amp_spectra = []  # Dùng để tính mặt nạ năng lượng tần số cao
        for c in range(C):
            x_c = x[:, c:c + 1, :, :]
            x_fft = torch.fft.fftshift(torch.fft.fft2(x_c))
            for idx in range(self.total_filters):
                H_sθ = self.freq_gabor_filters[idx:idx + 1, :, :, :]
                filtered_fft = x_fft * H_sθ
                amp_spectra.append(torch.abs(filtered_fft))
        amp_spectra = torch.cat(amp_spectra, dim=1)  # Sử dụng để tính mặt nạ tần số cao

        freq_feat = self._extract_freq_features(x)  # (B, C_f, H, W)

        # 3. Hướng dẫn chéo miền (sử dụng trọng số tần số cao)
        spatial_guided, freq_guided = self._cross_domain_guidance(spatial_feat, freq_feat, amp_spectra)

        # 4. Hợp nhất đặc trưng
        fused_feat = torch.cat([spatial_guided, freq_guided], dim=1)  # (B, C_s+C_f, H, W)
        fused_feat = self.fusion_conv(fused_feat)  # (B, 128, H, W)

        return fused_feat


# Kiểm tra mô-đun GSFTM
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")

    # Kiểm tra mô-đun GSFTM (bao gồm tham số trọng số tần số cao)
    gsftm = GSFTM(
        image_size=(224, 224),
        num_scales=2,
        num_orientations=4,
        high_freq_weight=0.8  # Tham số kiểm tra truyền vào
    ).to(device)
    x = torch.randn(2, 3, 224, 224).to(device)  # (B, C, H, W)
    fused = gsftm(x)
    print(f"Đầu ra GSFTM: {fused.shape}")  # nên dùng (2, 128, 224, 224)