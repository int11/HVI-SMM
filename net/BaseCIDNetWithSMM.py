import torch.nn as nn
from .BaseCIDNet import BaseCIDNet
from huggingface_hub import PyTorchModelHubMixin

class SMM(nn.Module):
    # Spatial Modulation Module(SMM)
    def __init__(self, in_channels, gamma=0.5):
        super(SMM, self).__init__()
        self.predictor = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, in_channels // 2, 3, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, in_channels // 2),
            nn.SiLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels // 2, 2, 3, stride=1, padding=0, bias=False),  # output 2 channels for alpha_s, alpha_i
            nn.Tanh()  # Tanh 출력 범위: [-1, 1]
        )
        self.gamma = gamma  # Tanh 스케일링 파라미터 (예: 0.5이면 범위 [0.5, 1.5])

    def forward(self, x):
        """
        Predict scale_factor from input features.
        
        Args:
            x: Input features
            
        Returns:
            scale_factor: (batch, 2, h, w) - scale factors for S and I channels
        """
        alpha_maps = self.predictor(x)  # Tanh 출력: [-1, 1]

        # [새로운 방식] Tanh(x) * gamma + 1
        # gamma = 0.5이면 범위: [-0.5 + 1, 0.5 + 1] = [0.5, 1.5]
        # CIDNet이 태업하지 못하고 최소한 원본에 근접한 출력을 내야 함
        # SSM은 미세 조정만 가능
        scale_factor = alpha_maps * self.gamma + 1.0
        return scale_factor


class BaseCIDNet_SMM(BaseCIDNet, PyTorchModelHubMixin):
    def set_alpha_predict(self, alpha_predict):
        """Set whether to predict alpha values"""
        self.alpha_predict = alpha_predict
    
    def forward_features(self, x):
        """
        Extract intermediate features without applying final alpha scaling.
        Returns features needed for multiple alpha combinations.
        
        Returns:
            output_hvi: HVI representation after encoder-decoder
            scale_factor: Predicted scale factors from SMM (base=1.0)
        """
        raise NotImplementedError("Subclass must implement forward_features()")
        return output_rgb