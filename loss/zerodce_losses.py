import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorConstancyLoss(nn.Module):
    def __init__(self):
        super(ColorConstancyLoss, self).__init__()

    def forward(self, x):
        """Gray-World assumption에 기반하여 각 채널의 평균이 동일해지도록 강제"""
        mean_rgb = torch.mean(x, [2, 3], keepdim=True) # (B, 3, 1, 1)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        d_rg = torch.pow(mr - mg, 2)
        d_rb = torch.pow(mr - mb, 2)
        d_gb = torch.pow(mb - mg, 2)
        loss = torch.pow(torch.pow(d_rg, 2) + torch.pow(d_rb, 2) + torch.pow(d_gb, 2), 0.5)
        return loss.mean()

class ExposureControlLoss(nn.Module):
    def __init__(self, patch_size=16, mean_val=0.6):
        super(ExposureControlLoss, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        """로컬 패치의 평균 밝기가 목적(E=0.6)에 가까워지도록 강제"""
        x_mean = torch.mean(x, 1, keepdim=True) # (B, 1, H, W)
        mean_pooled = self.pool(x_mean)
        loss = torch.mean(torch.pow(mean_pooled - self.mean_val, 2))
        return loss

class IlluminationSmoothnessLoss(nn.Module):
    def __init__(self):
        super(IlluminationSmoothnessLoss, self).__init__()

    def forward(self, x):
        """TV(Total Variation) 패턴을 통해 조명 맵이나 출력 결과가 부분적으로 부드럽게 유지되도록 만듦"""
        B, C, H, W = x.size()
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :H - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :W - 1]), 2).sum()
        return 2 * (h_tv / ((H - 1) * W) + w_tv / (H * (W - 1))) / B

class SpatialConsistencyLoss(nn.Module):
    def __init__(self):
        super(SpatialConsistencyLoss, self).__init__()
        # 인접한 4방향 이웃과의 차이를 구하기 위한 커널
        kernel_left  = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up    = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down  = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)
        
        self.weight_left  = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up    = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down  = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        """원본 이미지와 향상된 이미지 간의 이웃 픽셀 변화량 비율을 일치시켜 공간적 일관성 확보"""
        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)
        
        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        device = org.device
        weight_left = self.weight_left.to(device)
        weight_right = self.weight_right.to(device)
        weight_up = self.weight_up.to(device)
        weight_down = self.weight_down.to(device)

        D_org_left = F.conv2d(org_pool, weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, weight_down, padding=1)

        D_enhance_left = F.conv2d(enhance_pool, weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, weight_down, padding=1)

        D_left = torch.pow(D_org_left - D_enhance_left, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)

        return (D_left + D_right + D_up + D_down).mean()
