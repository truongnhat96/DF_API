import torch
import torch.nn as nn
import torch.nn.functional as F


class CSMIAM(nn.Module):
    """（Channel-Spatial Mutual Information Attention Module）"""

    def __init__(self):
        super(CSMIAM, self).__init__()

    def forward(self, x):
        # Hình dạng tính năng đầu vào: (B, C, H, W)
        # CRITICAL: Đảm bảo contiguous cho multi-GPU (DDP)
        x = x.contiguous()
        B, C, H, W = x.shape

        # 1. Tính toán thống kê kênh c_k (giá trị trung bình toàn cục của mỗi kênh)
        c = torch.mean(x, dim=(2, 3), keepdim=True)  # Hình dạng: (B, C, 1, 1)

        # 2. Tính toán thống kê không gian s_ij (giá trị trung bình qua các kênh tại mỗi vị trí không gian)
        s = torch.mean(x, dim=1, keepdim=True)  # Hình dạng: (B, 1, H, W)
        
        # 3. Tính toán ma trận tương tự cosine giữa thống kê kênh và thống kê không gian
        # Chuẩn hóa thống kê kênh, bỏ qua chuẩn hóa không gian kênh vì dim=1 sinh lỗi s_norm = 1.0
        eps = 1e-8
        c_norm = F.normalize(c, dim=1, eps=eps)  # Chuẩn hóa theo chiều kênh
        s_norm = s  # Giữ nguyên bản đồ không gian, softmax ở dưới sẽ lo việc cân bằng băng thông


        # Tính độ tương tự cosine: M_{k,i,j} = (c_k · s_ij) / (||c_k|| · ||s_ij||)
        # Thực hiện nhân phần tử theo cơ chế broadcast giữa (B, C, 1, 1) và (B, 1, H, W)
        M = c_norm * s_norm  # Hình dạng: (B, C, H, W)
        M = M.contiguous()  # CRITICAL: Ensure contiguous

        # 4. Tạo trọng số chú ý kênh A_c
        # Tổng theo chiều không gian rồi áp dụng Softmax
        channel_sum = torch.sum(M, dim=(2, 3))  # Hình dạng: (B, C)
        # Clamp để tránh numerical issues
        channel_sum = torch.clamp(channel_sum, min=-50, max=50)
        A_c = F.softmax(channel_sum, dim=1).unsqueeze(2).unsqueeze(3)  # Hình dạng: (B, C, 1, 1)

        # 5. Tạo trọng số chú ý không gian A_s
        # Tổng theo chiều kênh rồi áp dụng Softmax (thực hiện hai lần để tương thích với các phiên bản PyTorch cũ)
        spatial_sum = torch.sum(M, dim=1)  # Hình dạng: (B, H, W)
        # Clamp để tránh numerical issues
        spatial_sum = torch.clamp(spatial_sum, min=-50, max=50)
        # Trước tiên áp dụng softmax theo chiều W, sau đó theo chiều H
        A_s = F.softmax(F.softmax(spatial_sum, dim=2), dim=1).unsqueeze(1)  # Hình dạng: (B, 1, H, W)

        # 6. Kết hợp trọng số chú ý với đặc trưng gốc
        x_att = x * A_c * A_s  # Hình dạng: (B, C, H, W)
        # CRITICAL: Ensure contiguous before return
        x_att = x_att.contiguous()
        return x_att


class MultiScaleCSMIAM(nn.Module):
    """Mô-đun chú ý thông tin tương hỗ không gian kênh đa quy mô"""

    def __init__(self, in_channels, kernel_sizes=[1, 3, 5]):
        super(MultiScaleCSMIAM, self).__init__()
        self.kernel_sizes = kernel_sizes

        # Xây dựng các nhánh tích chập đa quy mô
        self.scale_branches = nn.ModuleList()
        for k in kernel_sizes:
            # Mỗi nhánh bao gồm: lớp tích chập + mô-đun CSMIAM
            branch = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=k,
                    padding=k // 2,  # Giữ nguyên kích thước không gian
                    groups=in_channels  # Tích chập phân tách kênh, giảm số tham số
                ),
                CSMIAM()
            )
            self.scale_branches.append(branch)

        # Tích hợp các đặc trưng đa quy mô bằng tích chập 1x1
        self.fusion_conv = nn.Conv2d(
            in_channels * len(kernel_sizes),  # Số kênh sau khi ghép nối
            in_channels,  # Số kênh đầu ra giữ nguyên
            kernel_size=1
        )

        # Chuẩn hóa hàng loạt các phép nối và kết nối dư
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        # Bảo toàn đặc trưng gốc cho kết nối dư
        x_res = x

        # Trích xuất đặc trưng từ các nhánh đa quy mô
        scale_features = []
        for branch in self.scale_branches:
            feat = branch(x)
            scale_features.append(feat)

        # Ghép nối và tích hợp đặc trưng đa quy mô
        fused = torch.cat(scale_features, dim=1)  # Ghép nối theo chiều kênh
        fused = self.fusion_conv(fused)  # Tích hợp thành số kênh ban đầu

        # Kết nối dư + kích hoạt
        out = self.bn(fused + x_res)
        out = F.relu(out)

        return out


# Mã kiểm tra
if __name__ == '__main__':
    # Kiểm tra xem có thiết bị CUDA khả dụng không
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")
    # Mô phỏng đầu vào đặc trưng (kích thước batch=2, số kênh=64, kích thước bản đồ đặc trưng=32x32)
    x = torch.randn(2, 64, 32, 32).to(device)

    # Kiểm tra mô-đun CSMIAM cơ bản
    csmiam = CSMIAM().to(device)
    out1 = csmiam(x)
    print(f"Hình dạng đầu ra CSMIAM: {out1.shape}")  # Nên giữ nguyên (2, 64, 32, 32)

    # Kiểm tra mô-đun MultiScaleCSMIAM
    multi_scale_csmiam = MultiScaleCSMIAM(in_channels=64).to(device)
    out2 = multi_scale_csmiam(x)
    print(f"Hình dạng đầu ra MultiScaleCSMIAM: {out2.shape}")  # Nên giữ nguyên (2, 64, 32, 32)
