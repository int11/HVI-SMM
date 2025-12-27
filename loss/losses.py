import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.vgg_arch import VGGFeatureExtractor, Registry
from loss.loss_utils import *
import dist

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
        gt_features = self.vgg(gt.detach())

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
    - RGB/HVI 공간에서 L1, SSIM, Edge, Perceptual loss 계산
    - Intermediate Supervision (I_base) 지원
    - 중복 코드 제거 및 깔끔한 인터페이스 제공
    """
    def __init__(self, 
                 rgb_to_hvi_fn,
                 L1_weight=1.0,
                 D_weight=0.5,
                 E_weight=50.0,
                 P_weight=1e-2,
                 HVI_weight=1.0,
                 intermediate_weight=0.5):
        super(CIDNetCombinedLoss, self).__init__()
        self.rgb_to_hvi_fn = rgb_to_hvi_fn
        self.HVI_weight = HVI_weight
        self.intermediate_weight = intermediate_weight
        
        # Loss 객체들을 내부에서 생성
        self.L1_loss = L1Loss(loss_weight=L1_weight, reduction='mean')
        self.D_loss = SSIM(weight=D_weight)
        self.E_loss = EdgeLoss(loss_weight=E_weight)
        # PerceptualLoss는 내부 weight를 P_weight로 설정 (외부에서 별도로 곱하지 않음)
        self.P_loss = PerceptualLoss(
            {'conv1_2': 1, 'conv2_2': 1, 'conv3_4': 1, 'conv4_4': 1}, 
            perceptual_weight=P_weight,  # P_weight를 여기서 적용
            criterion='mse'
        )
    
    def _compute_single_loss(self, pred_rgb, gt_rgb, gt_hvi):
        """
        하나의 RGB 예측값에 대해 RGB/HVI loss 계산
        
        Args:
            pred_rgb: 예측 RGB 이미지
            gt_rgb: GT RGB 이미지
            gt_hvi: GT HVI 이미지 (미리 계산됨)
        
        Returns:
            total_loss: RGB loss + HVI_weight * HVI loss
        """
        # RGB space loss
        loss_rgb = (self.L1_loss(pred_rgb, gt_rgb) + 
                   self.D_loss(pred_rgb, gt_rgb) + 
                   self.E_loss(pred_rgb, gt_rgb) + 
                   self.P_loss(pred_rgb, gt_rgb)[0])  # P_weight는 이미 내부에 적용됨
        
        # HVI space loss
        pred_hvi = self.rgb_to_hvi_fn(pred_rgb)
        loss_hvi = (self.L1_loss(pred_hvi, gt_hvi) + 
                   self.D_loss(pred_hvi, gt_hvi) + 
                   self.E_loss(pred_hvi, gt_hvi) + 
                   self.P_loss(pred_hvi, gt_hvi)[0])  # P_weight는 이미 내부에 적용됨
        
        return loss_rgb + self.HVI_weight * loss_hvi
    
    def forward(self, output_rgb, output_rgb_base, gt_rgb):
        """
        CIDNet SSM의 최종 loss 계산
        
        Args:
            output_rgb: SSM이 적용된 최종 출력 (I_out)
            output_rgb_base: SSM 없이 기본 alpha만 사용한 출력 (I_base)
            gt_rgb: Ground truth RGB
        
        Returns:
            total_loss: 최종 loss (final + intermediate)
        """
        # GT HVI는 한 번만 계산
        gt_hvi = self.rgb_to_hvi_fn(gt_rgb)
        
        # Final output loss (I_out)
        loss_final = self._compute_single_loss(output_rgb, gt_rgb, gt_hvi)
        
        # Intermediate supervision loss (I_base)
        if self.intermediate_weight > 0:
            loss_intermediate = self._compute_single_loss(output_rgb_base, gt_rgb, gt_hvi)
            total_loss = loss_final + self.intermediate_weight * loss_intermediate
        else:
            total_loss = loss_final
        
        return total_loss


