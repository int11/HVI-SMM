import torch
import torch.nn as nn
from net.BaseCIDNetWithSMM import BaseCIDNet_SMM, SMM
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub import hf_hub_download
import safetensors.torch as sf


class CIDNet_SMM(BaseCIDNet_SMM, PyTorchModelHubMixin):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False,
                 cidnet_model_path="Fediory/HVI-CIDNet-LOLv1-woperc",
                 sam_model_path="Gourieff/ReActor/models/sams/sam_vit_b_01ec64.pth",
                 gamma=0.5
        ):
        super(CIDNet_SMM, self).__init__()

        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0,bias=False)
            )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm = norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm = norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm = norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0,bias=False)
        )
        
        
        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0,bias=False),
            )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0,bias=False),
            )
        
        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)
        
        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)
        
        self.trans = RGB_HVI()
        self.alpha_predictor = SMM(in_channels=ch2*2, gamma=gamma)
    
    def forward_features(self, x):
        """
        Extract features without final alpha scaling.
        This method computes output_hvi and scale_factor which can be reused
        for multiple alpha combinations without redundant forward passes.
        """
        dtypes = x.dtype
        hvi = self.trans.RGB_to_HVI(x)
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)
        # low
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0 = i_enc0
        hv_jump0 = hv_0
        
        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)
        
        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc3)
        hv_3 = self.HVE_block3(hv_3)
        
        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)
        


        
        i_dec3 = self.I_LCA4(i_enc4,hv_4)
        hv_3 = self.HV_LCA4(hv_4, i_enc4)
        
        hv_3 = self.HVD_block3(hv_3, hv_jump2)
        i_dec3 = self.ID_block3(i_dec3, v_jump2)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)
        
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec2, v_jump1)
        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)
        
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)
        
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        

        alpha_input = torch.cat([i_dec1, hv_1], dim=1)
        scale_factor = self.alpha_predictor(alpha_input)

        return output_hvi, scale_factor
    
    def forward(self, x):
        """
        Forward pass with optional alpha prediction.
        Refactored to use forward_features() to avoid code duplication.
        """
        # Extract features once
        output_hvi, scale_factor = self.forward_features(x)
        
        # Base output (intermediate supervision)
        output_rgb_base = self.apply_alpha_scaling(
            output_hvi, None,
            self.base_alpha_s, self.base_alpha_i, self.alpha_rgb
        )
        
        output_rgb = self.apply_alpha_scaling(
            output_hvi, scale_factor,
            self.base_alpha_s, self.base_alpha_i, self.alpha_rgb
        )

        # CIDNet_SSM always return (output_rgb, output_rgb_base)
        return output_rgb, output_rgb_base
    

if __name__ == "__main__":
    model = CIDNet_SMM()
    model.eval()
    dummy_input = torch.randn(2, 3, 400, 400)  # Example input tensor
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Should be (2, 3, 400, 400) for RGB output
    print("Model loaded and tested successfully.")