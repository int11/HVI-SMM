import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.vgg_arch import VGGFeatureExtractor
import scripts.dist as dist

class RaGANLoss(nn.Module):
    """
    Relativistic Average GAN loss (LSGAN version)
    As used in EnlightenGAN paper.
    """
    def __init__(self):
        super(RaGANLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target_is_real, is_discriminator=True):
        # In RaGAN, the input 'pred' is usually (pred_real - mean(pred_fake)) or (pred_fake - mean(pred_real))
        if target_is_real:
            return self.mse(pred, torch.ones_like(pred))
        else:
            return self.mse(pred, torch.zeros_like(pred))

class SelfFeaturePreservingLoss(nn.Module):
    """
    Self Feature Preserving Loss (SFP) using VGG-16 features.
    Includes Instance Normalization before MSE as per EnlightenGAN.
    """
    def __init__(self, vgg_type='vgg16', layer_name='relu5_1', use_instance_norm=True):
        super(SelfFeaturePreservingLoss, self).__init__()
        self.vgg = VGGFeatureExtractor(
            layer_name_list=[layer_name],
            vgg_type=vgg_type,
            use_input_norm=True, # Option A: Use standard ImageNet normalization
            range_norm=False
        )
        self.use_instance_norm = use_instance_norm
        if use_instance_norm:
            # relu5_1 has 512 channels in VGG16
            self.instancenorm = nn.InstanceNorm2d(512, affine=False)

    def forward(self, pred, target):
        # ImageNet normalization is now handled inside VGGFeatureExtractor.
        # It expects input in [0, 1] RGB, which is our current format.
        pred_fea = self.vgg(pred)[self.vgg.layer_name_list[0]]
        target_fea = self.vgg(target)[self.vgg.layer_name_list[0]]
        
        if self.use_instance_norm:
            return F.mse_loss(self.instancenorm(pred_fea), self.instancenorm(target_fea))
        else:
            return F.mse_loss(pred_fea, target_fea)

class GANLoss(nn.Module):
    """Standard GAN loss (LSGAN or BCE)"""
    def __init__(self, use_lsgan=True):
        super(GANLoss, self).__init__()
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target_is_real):
        if target_is_real:
            target = torch.ones_like(pred)
        else:
            target = torch.zeros_like(pred)
        return self.loss(pred, target)


class FoFADiscriminatorLoss(nn.Module):
    """
    FoFA 학습용 판별기 + 생성기 손실 모듈.

    논문 Eq.(8)(9) 구현:
    - D 장데: real high 이미지를 real로, G(low) 를 fake로 판별하게 학습
    - D feature matching: D 편 특징 맵이 foundation feature와 정렬되도록 학습
    - G adversarial: D가 G(low)를 real로 볼 수 있도록 학습
    """
    def __init__(self, lambda_feat: float = 1.0):
        """
        Args:
            lambda_feat: feature matching loss 가중치
        """
        super(FoFADiscriminatorLoss, self).__init__()
        self.lambda_feat = lambda_feat

    def loss_D(self, pred_real: torch.Tensor, pred_fake: torch.Tensor,
               feat_real: torch.Tensor, foundation_real: torch.Tensor,
               feat_fake: torch.Tensor, foundation_fake: torch.Tensor) -> torch.Tensor:
        """
        판별기 업데이트 손실 (Eq.8).

        Args:
            pred_real: D(high_img) 로짃 (B,1,H,W) 또는 (B,1)
            pred_fake: D(G(low)) 로짃
            feat_real: D 내부 feature (high 이미지)
            foundation_real: foundation model projected feature (high 이미지)
            feat_fake: D 내부 feature (G(low) 이미지)
            foundation_fake: foundation model projected feature (G(low) 이미지)

        Returns:
            loss_D scalar
        """
        # adversarial loss (BCE with logits)
        adv_real = F.binary_cross_entropy_with_logits(
            pred_real, torch.ones_like(pred_real)
        )
        adv_fake = F.binary_cross_entropy_with_logits(
            pred_fake, torch.zeros_like(pred_fake)
        )
        # feature alignment loss
        feat_align_real = F.mse_loss(feat_real, foundation_real.detach())
        feat_align_fake = F.mse_loss(feat_fake, foundation_fake.detach())

        return adv_real + adv_fake + self.lambda_feat * (feat_align_real + feat_align_fake)

    def loss_G(self, pred_fake: torch.Tensor) -> torch.Tensor:
        """
        생성기 adversarial 손실 (Eq.9).

        Args:
            pred_fake: D(G(low)) 로짃

        Returns:
            loss_G_adv scalar
        """
        return F.binary_cross_entropy_with_logits(
            pred_fake, torch.ones_like(pred_fake)
        )
