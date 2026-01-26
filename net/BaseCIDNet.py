import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

class BaseCIDNet(nn.Module, PyTorchModelHubMixin):
    def set_base_alpha(self, base_alpha_s=1.0, base_alpha_i=1.0, alpha_rgb=1.0):
        """Set base alpha values for S, I channels and RGB scaling"""
        self.base_alpha_s = base_alpha_s
        self.base_alpha_i = base_alpha_i
        self.alpha_rgb = alpha_rgb

    def RGB_to_HVI(self, x):
        hvi = self.trans.RGB_to_HVI(x)
        return hvi