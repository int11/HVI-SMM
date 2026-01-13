import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.vgg_arch import VGGFeatureExtractor, Registry
from loss.loss_utils import *
import scripts.dist as distpts.dist as dist

_reduction_modes = ['none', 'mean', 'sum']

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)
         
        
        
class EdgeLoss(nn.Module):
    def __init__(self,loss_weight=1.0, reduction='mean'):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1).to(dist.get_device())

        self.weight = loss_weight
        
    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = mse_loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss*self.weight


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=True,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='mean')
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def extract_features(self, x):
        """Extract VGG features (for caching GT features)."""
        with torch.no_grad():
            return self.vgg(x)
    
    def forward_with_gt_features(self, x, gt_features):
        """Forward with pre-computed GT features (faster)."""
        x_features = self.vgg(x)
        return self._compute_loss(x_features, gt_features)
    
    def _compute_loss(self, x_features, gt_features):
        """Compute perceptual and style loss from features."""
        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        with torch.no_grad():
            gt_features = self.vgg(gt.detach())
        
        return self._compute_loss(x_features, gt_features)




class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True,weight=1.):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.weight = weight

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return (1. - map_ssim(img1, img2, window, self.window_size, channel, self.size_average)) * self.weight


class CIDNetCombinedLoss(nn.Module):
    """
    CIDNet SSM을 위한 통합 Loss 클래스
    """
    def __init__(self, 
                 trans,
                 L1_weight=1.0,
                 D_weight=0.5,
                 E_weight=50.0,
                 P_weight=1e-2,
                 HVI_weight=1.0,
                 use_gt_mean_loss='rgb',
                 sigma=0.1):
        super(CIDNetCombinedLoss, self).__init__()
        self.trans = trans
        self.HVI_weight = HVI_weight
        self.use_gt_mean_loss = use_gt_mean_loss
        self.sigma = sigma
        
        # Loss 객체들
        self.L1_loss = L1Loss(loss_weight=L1_weight, reduction='mean')
        self.D_loss = SSIM(weight=D_weight)
        self.E_loss = EdgeLoss(loss_weight=E_weight)
        self.P_loss = PerceptualLoss(
            {'conv1_2': 1, 'conv2_2': 1, 'conv3_4': 1, 'conv4_4': 1}, 
            perceptual_weight=P_weight,
            criterion='mse'
        )

        # Grayscale 변환 weight (밝기 평균 계산용)
        self.register_buffer('grayscale_weight', torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

        # Loss 함수 선택 (초기화 시 한 번만 결정)
        if self.use_gt_mean_loss == 'rgb':
            self.loss_func = self._compute_gt_mean_loss
        elif self.use_gt_mean_loss == 'hvi':
            self.loss_func = self._compute_intensity_mean_loss
        else:  # 'none'
            self.loss_func = self._compute_basic_loss

    def get_brightness_mean(self, img):
        # img: (N, 3, H, W) -> (N,)
        gray = F.conv2d(img, self.grayscale_weight, bias=None)
        return torch.mean(gray, dim=(2, 3)).squeeze(1)

    def KL_div(self, mu_1, sigma_1, mu_2, sigma_2):
        """
        가우시안 분포 간의 KL Divergence 계산.
        
        두 정규분포 P(mu_1, sigma_1)과 Q(mu_2, sigma_2) 간의 KL Divergence D_KL(P || Q)를 계산합니다.
        
        Args:
            mu_1, sigma_1: P 분포(첫 번째 분포)의 평균과 표준편차
            mu_2, sigma_2: Q 분포(두 번째 분포)의 평균과 표준편차
        
        Returns:
            D_KL(P || Q) = log(sigma_2/sigma_1) + (sigma_1^2 + (mu_1-mu_2)^2)/(2*sigma_2^2) - 0.5
        """
        return torch.log(sigma_2 / sigma_1) + 0.5 * (sigma_1**2 + (mu_1 - mu_2)**2) / sigma_2**2 - 0.5

    def _compute_basic_loss(self, pred_rgb, pred_hvi, gt_rgb, gt_hvi, gt_features_rgb, gt_features_hvi):
        """기존 CIDNet의 Loss 계산 (L1 + SSIM + Edge + Perceptual)"""
        # RGB space loss
        loss_rgb = (self.L1_loss(pred_rgb, gt_rgb) + 
                   self.D_loss(pred_rgb, gt_rgb) + 
                   self.E_loss(pred_rgb, gt_rgb) + 
                   self.P_loss.forward_with_gt_features(pred_rgb, gt_features_rgb)[0])
        
        # HVI space loss
        loss_hvi = (self.L1_loss(pred_hvi, gt_hvi) + 
                   self.D_loss(pred_hvi, gt_hvi) + 
                   self.E_loss(pred_hvi, gt_hvi) + 
                   self.P_loss.forward_with_gt_features(pred_hvi, gt_features_hvi)[0])
        
        return loss_rgb + self.HVI_weight * loss_hvi

    def _compute_gt_mean_loss(self, pred_rgb, pred_hvi, gt_rgb, gt_hvi, gt_features_rgb, gt_features_hvi):
        """
        GT-Mean Loss 적용 (RGB 밝기 기반):
        밝기 평균(mu)을 한 번만 계산하여 Weight(W)와 Scaling Factor(m)에 모두 사용
        """
        # 1. 밝기 평균 계산
        mu_gt = self.get_brightness_mean(gt_rgb)   # E[y]
        mu_pred = self.get_brightness_mean(pred_rgb) # E[f(x)]

        # --- [A] Weight W 계산 ---
        # 0으로 나누기 방지 및 안전한 계산
        mu_gt_safe = torch.abs(mu_gt) + 1e-8
        mu_pred_safe = torch.abs(mu_pred) + 1e-8

        sigma_gt = self.sigma * mu_gt_safe
        sigma_pred = self.sigma * mu_pred_safe
        
        mu_M = 0.5 * (mu_gt_safe + mu_pred_safe)
        sigma_M = torch.sqrt((sigma_gt**2 + sigma_pred**2) / 2)
        
        # 양방향 KL Divergence 계산 (Symmetric)
        kl_val = 0.5 * self.KL_div(mu_gt_safe, sigma_gt, mu_M, sigma_M) + \
                 0.5 * self.KL_div(mu_pred_safe, sigma_pred, mu_M, sigma_M)
                 
        # Weight clipping [0, 1] & Broadcasting shape 준비
        W = torch.clamp(kl_val, 0, 1).view(-1, 1, 1, 1)
        # -------------------------------

        # --- [B] Scaling Factor m 계산 (위에서 구한 mu 재사용) ---
        m = mu_gt / (mu_pred + 1e-8)
        m = m.view(-1, 1, 1, 1)
        
        # 2. Brightness Aligned Prediction 생성
        pred_aligned_rgb = torch.clamp(m * pred_rgb, 0, 1)
        pred_aligned_hvi = self.trans.RGB_to_HVI(pred_aligned_rgb)
        
        # 3. Loss 계산
        # Original Term: L(f(x), y)
        loss_orig = self._compute_basic_loss(pred_rgb, pred_hvi, gt_rgb, gt_hvi, 
                                             gt_features_rgb, gt_features_hvi)
        
        # Aligned Term: L(m * f(x), y)
        loss_aligned = self._compute_basic_loss(pred_aligned_rgb, pred_aligned_hvi, gt_rgb, gt_hvi,
                                                gt_features_rgb, gt_features_hvi)
        
        # 4. 최종 결합: W * L_orig + (1-W) * L_aligned
        # 배치 내 평균 Weight 사용 (전체 Loss reduction이 mean이므로)
        W_mean = torch.mean(W)
        total_loss = W_mean * loss_orig + (1 - W_mean) * loss_aligned
        
        return total_loss
    
    def _compute_intensity_mean_loss(self, pred_rgb, pred_hvi, gt_rgb, gt_hvi, gt_features_rgb, gt_features_hvi):
        """
        Intensity Mean Loss 적용 (HVI Intensity 채널 기반):
        HVI의 intensity 채널에서 밝기 평균을 계산하여 GT-Mean Loss 적용
        """
        # 1. HVI Intensity 채널 추출 (I channel: index 2)
        gt_intensity = gt_hvi[:, 2, :, :]  # (N, H, W)
        pred_intensity = pred_hvi[:, 2, :, :]  # (N, H, W)
        
        # 2. Intensity 채널에서 평균 계산
        mu_gt = torch.mean(gt_intensity, dim=(1, 2))  # (N,)
        mu_pred = torch.mean(pred_intensity, dim=(1, 2))  # (N,)

        # --- [A] Weight W 계산 ---
        # 0으로 나누기 방지 및 안전한 계산
        mu_gt_safe = torch.abs(mu_gt) + 1e-8
        mu_pred_safe = torch.abs(mu_pred) + 1e-8

        sigma_gt = self.sigma * mu_gt_safe
        sigma_pred = self.sigma * mu_pred_safe
        
        mu_M = 0.5 * (mu_gt_safe + mu_pred_safe)
        sigma_M = torch.sqrt((sigma_gt**2 + sigma_pred**2) / 2)
        
        # 양방향 KL Divergence 계산 (Symmetric)
        kl_val = 0.5 * self.KL_div(mu_gt_safe, sigma_gt, mu_M, sigma_M) + \
                 0.5 * self.KL_div(mu_pred_safe, sigma_pred, mu_M, sigma_M)
                 
        # Weight clipping [0, 1] & Broadcasting shape 준비
        W = torch.clamp(kl_val, 0, 1).view(-1, 1, 1, 1)
        # -------------------------------

        # --- [B] Scaling Factor m 계산 (Intensity 채널 기반) ---
        m = mu_gt / (mu_pred + 1e-8)
        m = m.view(-1, 1, 1, 1)
        
        # 2. HVI 공간에서 Intensity-Aligned 생성
        pred_aligned_hvi = pred_hvi.clone()
        pred_aligned_hvi[:, 2, :, :] = torch.clamp(m.squeeze(1) * pred_intensity, 0, 1)
        
        # 3. HVI Aligned를 RGB로 변환
        pred_aligned_rgb = self.trans.HVI_to_RGB(pred_aligned_hvi)
        
        # 4. Loss 계산
        # Original Term: L(f(x), y)
        loss_orig = self._compute_basic_loss(pred_rgb, pred_hvi, gt_rgb, gt_hvi,
                                             gt_features_rgb, gt_features_hvi)
        
        # Aligned Term: L(aligned(f(x)), y)
        loss_aligned = self._compute_basic_loss(pred_aligned_rgb, pred_aligned_hvi, gt_rgb, gt_hvi,
                                                gt_features_rgb, gt_features_hvi)
        
        # 5. 최종 결합: W * L_orig + (1-W) * L_aligned
        # 배치 내 평균 Weight 사용
        W_mean = torch.mean(W)
        total_loss = W_mean * loss_orig + (1 - W_mean) * loss_aligned
        
        return total_loss
    
    def forward(self, pred_rgb, gt_rgb, gt_features_rgb=None, gt_features_hvi=None):
        """
        Args:
            pred_rgb: 예측 이미지
            gt_rgb: Ground truth RGB
            gt_features_rgb: Pre-computed VGG features for gt_rgb (optional)
            gt_features_hvi: Pre-computed VGG features for gt_hvi (optional)
        """
        # GT 변환
        gt_hvi = self.trans.RGB_to_HVI(gt_rgb)
        pred_hvi = self.trans.RGB_to_HVI(pred_rgb)
        
        # GT features 계산 (None이면 새로 계산)
        if gt_features_rgb is None:
            gt_features_rgb = self.P_loss.extract_features(gt_rgb)
        if gt_features_hvi is None:
            gt_features_hvi = self.P_loss.extract_features(gt_hvi)
        
        return self.loss_func(pred_rgb, pred_hvi, gt_rgb, gt_hvi, gt_features_rgb, gt_features_hvi)


class CIDNetWithIntermediateLoss(nn.Module):
    """
    CIDNetCombinedLoss에 Intermediate Supervision을 추가하는 래퍼 클래스
    최적화: GT features를 캐싱하여 VGG forward 횟수 감소
    """
    def __init__(self, 
                 base_loss_fn,
                 intermediate_weight=0.5):
        """
        Args:
            base_loss_fn: CIDNetCombinedLoss 인스턴스
            intermediate_weight: Intermediate loss의 가중치
        """
        super(CIDNetWithIntermediateLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.intermediate_weight = intermediate_weight

    def forward(self, pred_rgb_final, pred_rgb_base, gt_rgb):
        """
        Args:
            pred_rgb_final: SSM이 적용된 최종 예측 이미지
            pred_rgb_base: SSM 없이 기본 alpha만 사용한 예측 이미지
            gt_rgb: Ground truth RGB
        """
        assert pred_rgb_base is not None, "pred_rgb_base must be provided for intermediate supervision"
        
        # GT features 캐싱 (한 번만 계산)
        gt_hvi = self.base_loss_fn.trans.RGB_to_HVI(gt_rgb)
        gt_features_rgb = self.base_loss_fn.P_loss.extract_features(gt_rgb)
        gt_features_hvi = self.base_loss_fn.P_loss.extract_features(gt_hvi)
        
        # Final output loss (캐싱된 GT features 사용)
        loss_final = self.base_loss_fn(pred_rgb_final, gt_rgb, gt_features_rgb, gt_features_hvi)
        
        # Intermediate supervision loss (캐싱된 GT features 재사용)
        loss_intermediate = self.base_loss_fn(pred_rgb_base, gt_rgb, gt_features_rgb, gt_features_hvi)
        
        # 가중 합
        total_loss = loss_final + self.intermediate_weight * loss_intermediate
        
        return total_loss