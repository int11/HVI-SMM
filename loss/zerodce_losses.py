import torch
import torch.nn as nn
import torch.nn.functional as F


def _intensity_map(x):
    """Max-RGB intensity map, matching HVI_transform.py RGB_to_HVI definition."""
    return x.amax(dim=1, keepdim=True)


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


# ---------------------------------------------------------------------------
# HVI-aware prior losses (I-map 기반으로 교체)
# ---------------------------------------------------------------------------

class IntensityExposureLoss(nn.Module):
    """ExposureControlLoss를 Max-RGB intensity 위에서 계산.

    Zero-DCE 원 구현의 채널평균 대신 I = max(R,G,B) 를 써서
    HVI 논문 §3.1 Max-RGB 정의와 정합.
    """
    def __init__(self, patch_size=16, mean_val=0.6):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        I = _intensity_map(x)                          # (B, 1, H, W)
        mean_pooled = self.pool(I)
        return torch.mean(torch.pow(mean_pooled - self.mean_val, 2))


class IntensitySpatialLoss(nn.Module):
    """SpatialConsistencyLoss를 Max-RGB intensity 위에서 계산.

    원본/향상 이미지 모두 I-map으로 추출해 4방향 gradient를 비교함으로써
    밝기 구조만 비교하고 색상 정보 유실을 피함.
    """
    def __init__(self):
        super().__init__()
        kernel_left  = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up    = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down  = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)
        self.weight_left  = nn.Parameter(data=kernel_left,  requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up    = nn.Parameter(data=kernel_up,    requires_grad=False)
        self.weight_down  = nn.Parameter(data=kernel_down,  requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        org_I     = self.pool(_intensity_map(org))
        enh_I     = self.pool(_intensity_map(enhance))
        device    = org.device
        wl, wr, wu, wd = (w.to(device) for w in
                          (self.weight_left, self.weight_right, self.weight_up, self.weight_down))
        D_org  = [F.conv2d(org_I, w, padding=1) for w in (wl, wr, wu, wd)]
        D_enh  = [F.conv2d(enh_I, w, padding=1) for w in (wl, wr, wu, wd)]
        return sum(torch.pow(do - de, 2) for do, de in zip(D_org, D_enh)).mean()


class IntensityTVLoss(nn.Module):
    """IlluminationSmoothnessLoss를 Max-RGB intensity 1채널에만 적용.

    sRGB 3채널 TV는 색 에지까지 뭉개는 문제가 있어,
    HV-branch가 학습한 색 구조를 보존하기 위해 I-map에만 걸음.
    주의: 1채널이라 동일 이미지 대비 값 스케일이 약 1/3.
    권장 lambda: 기존 lambda_tv=20 → 60으로 재보정.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        I = _intensity_map(x)                          # (B, 1, H, W)
        B, C, H, W = I.size()
        h_tv = torch.pow(I[:, :, 1:, :] - I[:, :, :H-1, :], 2).sum()
        w_tv = torch.pow(I[:, :, :, 1:] - I[:, :, :, :W-1], 2).sum()
        return 2 * (h_tv / ((H - 1) * W) + w_tv / (H * (W - 1))) / B


# ---------------------------------------------------------------------------
# HVI-native prior losses (HV-plane 활용)
# ---------------------------------------------------------------------------

class HVPreservationLoss(nn.Module):
    """HVI-native 색상 보존 prior.

    HVI 논문 §3, §4.3: C_k 함수가 밝기 의존성을 나눠주므로
    저조도 입력과 향상 결과의 HV 좌표는 상대적으로 안정적이어야 함.
    fake_high와 real_low의 HV-plane L1 거리를 최소화해 색 washout을 방지.

    trans: RGB_HVI 인스턴스 (netG.trans 재사용).
    """
    def __init__(self, trans):
        super().__init__()
        self.trans = trans

    def forward(self, fake_high, real_low):
        hv_fake = self.trans.RGB_to_HVI(fake_high)[:, :2]         # (B, 2, H, W)
        with torch.no_grad():
            hv_low  = self.trans.RGB_to_HVI(real_low)[:, :2]
        return F.l1_loss(hv_fake, hv_low)


class HVIIntensityTVLoss(nn.Module):
    """HVI 재투영 후 I-채널 TV (선택적).

    IntensityTVLoss와 거의 동치이나 완전히 HVI-space에서 계산.
    I_hvi = RGB_to_HVI(fake)[:, 2] 를 사용.
    기본 비활성(lambda_i_tv_hvi=0).
    """
    def __init__(self, trans):
        super().__init__()
        self.trans = trans

    def forward(self, x):
        I = self.trans.RGB_to_HVI(x)[:, 2:3]          # (B, 1, H, W)
        B, C, H, W = I.size()
        h_tv = torch.pow(I[:, :, 1:, :] - I[:, :, :H-1, :], 2).sum()
        w_tv = torch.pow(I[:, :, :, 1:] - I[:, :, :, :W-1], 2).sum()
        return 2 * (h_tv / ((H - 1) * W) + w_tv / (H * (W - 1))) / B
