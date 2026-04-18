import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


class DECAM(nn.Module):
    """Dual Entropy Cooperative Attention Module (DECAM)"""

    def __init__(self,
                 T_f=0.5,
                 T_p=0.3,
                 beta_init=0.5,  # Giảm giá trị khởi tạo để giảm rủi ro tràn số mũ
                 high_freq_weight=0.6,
                 center_ratio=0.6):  # Tỷ lệ vùng trung tâm
        super(DECAM, self).__init__()
        self.T_f = T_f  # Ngưỡng entropy đặc trưng
        self.T_p = T_p  # Ngưỡng entropy dự đoán
        self.beta = nn.Parameter(torch.tensor(beta_init))  # Trọng số có thể học được
        self.high_freq_weight = high_freq_weight  # Trọng số đặc trưng tần số cao
        self.center_ratio = center_ratio  # Tỷ lệ vùng trung tâm
        self.eps = 1e-6  # Giá trị bảo vệ tăng cường, cải thiện độ ổn định
        self.max_exp_input = 5.0  # Giới hạn đầu vào hàm mũ để tránh tràn số
        self.beta_max = 10.0  # Giới hạn giá trị tối đa của beta

    def _create_high_freq_mask(self, H, W, device):
        """Tạo mặt nạ tần số cao (tăng cường đặc trưng chi tiết hình ảnh)"""
        mask = torch.ones((H, W), device=device)
        cy, cx = H // 2, W // 2
        # Bán kính vùng tần số thấp: điều chỉnh động theo kích thước bản đồ đặc trưng
        r = min(cy, cx) // 4 if min(cy, cx) >= 4 else 1
        y, x = torch.meshgrid(torch.arange(H, device=device),
                              torch.arange(W, device=device),
                              indexing="xy")
        dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        # Giảm trọng số vùng tần số thấp, làm nổi bật đặc trưng tần số cao
        mask[dist <= r] = 1.0 - self.high_freq_weight
        return mask.unsqueeze(0).unsqueeze(0)  # (B,1,H,W)

    def calculate_feature_entropy(self, F):
        """Tính entropy đặc trưng (bao gồm tăng cường tần số cao, tối ưu độ ổn định số)"""
        B, C, H, W = F.shape
        device = F.device
        orig_dtype = F.dtype  # Lưu loại dữ liệu gốc

        # Kiểm tra xem có phải là half precision và kích thước không phải là lũy thừa của 2, nếu có thì chuyển sang float32 để tính FFT
        need_conversion = False
        if orig_dtype == torch.float16:
            # Kiểm tra tất cả các chiều không gian có phải là lũy thừa của 2 hay không
            if not ((H & (H - 1)) == 0 and H != 0 and
                    (W & (W - 1)) == 0 and W != 0):
                # Cần chuyển đổi sang float32 để tính FFT
                F = F.to(torch.float32)
                need_conversion = True

        # Biến đổi Fourier để trích xuất các đặc trưng tần số cao (có thêm biện pháp bảo vệ số học)
        F_fft = torch.fft.fft2(F)
        F_fft_shifted = torch.fft.fftshift(F_fft)

        # Loại bỏ các giá trị quang phổ cực đoan để ngăn ngừa hiện tượng tràn số sau khi biến đổi.
        mag = torch.abs(F_fft_shifted)
        mean_mag = torch.mean(mag)
        std_mag = torch.std(mag)
        clip_threshold = mean_mag + 5 * std_mag
        F_fft_shifted = torch.where(
            mag > clip_threshold,
            (clip_threshold) * torch.exp(1j * torch.angle(F_fft_shifted)),
            F_fft_shifted
        )

        # Áp dụng mặt nạ tần số cao
        high_freq_mask = self._create_high_freq_mask(H, W, device)
        F_fft_high = F_fft_shifted * high_freq_mask

        # Biến đổi Fourier ngược và giới hạn phạm vi giá trị
        F_high = torch.fft.ifftshift(F_fft_high)
        F_high = torch.fft.ifft2(F_high).real
        F_high = torch.clamp(F_high, min=-1e3, max=1e3)  # Ngăn ngừa giá trị cực đoan

        # Chuyển về loại dữ liệu gốc (nếu trước đó đã chuyển đổi)
        if need_conversion:
            F = F.to(orig_dtype)
            F_high = F_high.to(orig_dtype)

        # Kết hợp đặc trưng và chuẩn hóa (các bước ổn định quan trọng)
        F_fused = (F * (1 - self.high_freq_weight)) + (F_high * self.high_freq_weight)
        F_fused = F_fused / (torch.norm(F_fused, dim=1, keepdim=True) + self.eps)  # L2 chuẩn hóa

        # Tính entropy đặc trưng (tránh log(0))
        F_sq = F_fused ** 2
        F_norm_sq = torch.sum(F_sq, dim=1, keepdim=True)
        p_f = F_sq / (F_norm_sq + self.eps)
        p_f = torch.clamp(p_f, min=self.eps, max=1 - self.eps)  # Đảm bảo p_f trong khoảng (0,1)
        return -torch.sum(p_f * torch.log(p_f), dim=1)

    def calculate_prediction_entropy(self, P):
        """Tính entropy dự đoán (đảm bảo độ ổn định số)"""
        P_clamped = torch.clamp(P, min=self.eps, max=1 - self.eps)  # Ngăn ngừa P bằng 0 hoặc 1
        return -torch.sum(P_clamped * torch.log(P_clamped), dim=1)

    def forward(self, F, P):
        """Truyền tiến (áp dụng trọng số chú ý dựa trên hai entropy)"""
        B, C, H, W = F.shape
        device = F.device

        # 1. Tính entropy đặc trưng và entropy dự đoán
        H_f = self.calculate_feature_entropy(F)  # (B, H, W)
        H_p = self.calculate_prediction_entropy(P)  # (B, H, W)

        # 2. Khu vực trung tâm ưu tiên (tập trung vào khu vực trung tâm khuôn mặt)
        # center_size = int(min(H, W) * self.center_ratio)
        # start_h = max(0, (H - center_size) // 2)
        # end_h = start_h + center_size
        # start_w = max(0, (W - center_size) // 2)
        # end_w = start_w + center_size
        # center_mask = torch.zeros(B, H, W, device=device)
        # center_mask[:, start_h:end_h, start_w:end_w] = 1.0  # Đánh dấu khu vực trung tâm

        # 3. Lọc không chắc chắn phối hợp (lọc hai ngưỡng)
        # mask = (H_f >= self.T_f) & (H_p >= self.T_p) & (center_mask > 0)
        mask = (H_f >= self.T_f) & (H_p >= self.T_p)

        # 4. Tạo trọng số chú ý (giới hạn đầu vào hàm mũ)
        U = torch.sqrt(H_f * H_p + self.eps) * mask.float()
        max_U = torch.max(U.view(B, -1), dim=1, keepdim=True)[0].view(B, 1, 1)
        normalized_U = U / (max_U + self.eps)
        normalized_U = torch.clamp(normalized_U, max=self.max_exp_input)  # Giới hạn đầu vào hàm mũ
        # Giới hạn giá trị beta, tránh trọng số quá lớn
        beta_clamped = torch.clamp(self.beta, max=self.beta_max)
        A = torch.exp((beta_clamped * normalized_U).unsqueeze(1))  # (B,1,H,W)

        # 5. Trọng số đặc trưng
        return F * A, A.squeeze(1)


# --------------------------
# Hàm tiện ích đi kèm (tính ngưỡng)
# --------------------------
def filter_outliers(arr, upper_percentile=99):
    """Lọc các giá trị cao cực đoan, tăng cường độ ổn định khi tính ngưỡng"""
    if not arr:
        return arr
    threshold = np.percentile(arr, upper_percentile)
    return [x for x in arr if x <= threshold]


# tính toán ngưỡng DECAM với tùy chọn tập trung vào vùng trung tâm khuôn mặt
def calculate_tf_tp(model, train_dataset, sample_ratio=0.1, device='cuda', percentile=90, focus_face_core=False):
    """
    Tính toán ngưỡng entropy đặc trưng và entropy dự đoán của mô-đun DECAM để hỗ trợ việc tập trung vào vùng khuôn mặt cốt lõi.

    Args:
        model: mô hình đã được huấn luyện trước
        train_dataset: tập dữ liệu huấn luyện
        sample_ratio: tỷ lệ mẫu
        device: thiết bị tính toán
        percentile: phân vị
        focus_face_core: có chỉ sử dụng khu vực trung tâm khuôn mặt để tính ngưỡng (tham số mới)
    Returns:
        T_f: ngưỡng entropy đặc trưng
        T_p: ngưỡng entropy dự đoán
    """
    model.eval()
    # Chỉ lấy mẫu các mẫu thực để tính ngưỡng
    real_indices = [i for i, sample in enumerate(train_dataset.samples) if sample['label'] == 0]
    if not real_indices:
        raise ValueError("Không có mẫu thực trong tập huấn luyện để tính ngưỡng DECAM")

    # Lấy mẫu ngẫu nhiên từ các chỉ số thực
    sample_size = max(100, int(len(real_indices) * sample_ratio))
    sample_indices = np.random.choice(real_indices, size=sample_size, replace=False)

    Hf_list = []
    Hp_list = []

    with torch.no_grad():
        for idx in tqdm(sample_indices, desc="Tính ngưỡng DECAM"):
            image, _ = train_dataset[idx]
            image = image.unsqueeze(0).to(device)

            # Lấy đặc trưng và bản đồ xác suất
            _, features, prob_map = model.extract_features(image)  # Mô hình có phương thức này để trả về đặc trưng và bản đồ xác suất

            # Tính entropy đặc trưng và entropy dự đoán
            feat_softmax = F.softmax(features, dim=1)
            feat_logsoftmax = F.log_softmax(features, dim=1)
            H_f = -torch.sum(feat_softmax * feat_logsoftmax, dim=1).cpu().numpy()  # (1, H, W)

            # Tính entropy dự đoán
            prob_softmax = F.softmax(prob_map, dim=1)
            prob_logsoftmax = F.log_softmax(prob_map, dim=1)
            H_p = -torch.sum(prob_softmax * prob_logsoftmax, dim=1).cpu().numpy()  # (1, H, W)

            # Nếu tập trung vào vùng trung tâm khuôn mặt, chỉ giữ lại giá trị entropy của vùng trung tâm
            if focus_face_core:
                h, w = H_f.shape[1], H_f.shape[2]
                center_h, center_w = h // 2, w // 2
                region_size_h = int(h * 0.7 // 2)  # 70% khu vực trung tâm
                region_size_w = int(w * 0.7 // 2)

                # Trích xuất vùng trung tâm
                H_f_core = H_f[:,
                           center_h - region_size_h: center_h + region_size_h,
                           center_w - region_size_w: center_w + region_size_w]
                H_p_core = H_p[:,
                           center_h - region_size_h: center_h + region_size_h,
                           center_w - region_size_w: center_w + region_size_w]

                Hf_list.extend(H_f_core.flatten())
                Hp_list.extend(H_p_core.flatten())
            else:
                Hf_list.extend(H_f.flatten())
                Hp_list.extend(H_p.flatten())

    # Tính toán phân vị sau khi lọc các giá trị ngoại lai
    T_f = np.percentile(Hf_list, percentile)
    T_p = np.percentile(Hp_list, percentile)

    return T_f, T_p
