import torch
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
    
    def forward_features(self, x):
        """
        Extract intermediate features without applying final alpha scaling.
        Should return (output_hvi, scale_factor).
        
        For base CIDNet models, scale_factor is ones (no spatial modulation).
        For SMM models, scale_factor is predicted by the alpha predictor.
        
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclass must implement forward_features()")
    
    def apply_alpha_scaling(self, output_hvi, scale_factor, base_alpha_s=1.0, base_alpha_i=1.0, alpha_rgb=1.0):
        """
        Apply alpha scaling to pre-computed features.
        
        Args:
            output_hvi: Pre-computed HVI features
            scale_factor: Scale factors (batch, 2, h, w). None for base models, predicted tensor for SMM models.
            base_alpha_s, base_alpha_i, alpha_rgb: Scaling parameters
            
        Returns:
            output_rgb: Final RGB output
        """
        if scale_factor is None:
            # Base CIDNet path: direct alpha scaling without spatial modulation
            output_rgb = self.trans.HVI_to_RGB(output_hvi, base_alpha_s, base_alpha_i, alpha_rgb)
        else:
            # SMM path: scale_factor modulates alpha spatially
            alpha_s = base_alpha_s * scale_factor[:, 0, :, :]
            alpha_i = base_alpha_i * scale_factor[:, 1, :, :]
            output_rgb = self.trans.HVI_to_RGB(output_hvi, alpha_s, alpha_i, alpha_rgb)
        return output_rgb