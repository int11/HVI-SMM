import torch
import torch.nn as nn
from net.HVI_transform_SSM import RGB_HVI
from net.transformer_utils import *
from net.LCA import *
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub import hf_hub_download
import safetensors.torch as sf


class SSM(nn.Module):
    # Spatial Modulation Module(SSM)
    def __init__(self, in_channels, gamma=0.5):
        super(SSM, self).__init__()
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

    def forward(self, x, base_alpha_s=1.0, base_alpha_i=1.0):
        alpha_maps = self.predictor(x)  # Tanh 출력: [-1, 1]

        # [새로운 방식] Tanh(x) * gamma + 1
        # gamma = 0.5이면 범위: [-0.5 + 1, 0.5 + 1] = [0.5, 1.5]
        # CIDNet이 태업하지 못하고 최소한 원본에 근접한 출력을 내야 함
        # SSM은 미세 조정만 가능
        scale_factor = alpha_maps * self.gamma + 1.0

        # 예측된 스케일 팩터를 각 기준 알파 값에 곱해줍니다.
        alpha_s = base_alpha_s * scale_factor[:, 0, :, :]
        alpha_i = base_alpha_i * scale_factor[:, 1, :, :]
        return alpha_s, alpha_i  # alpha_s: (batch, 1, h, w), alpha_i: (batch, 1, h, w)


class CIDNet_SSM(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False,
                 cidnet_model_path="Fediory/HVI-CIDNet-LOLv1-woperc",
                 sam_model_path="Gourieff/ReActor/models/sams/sam_vit_b_01ec64.pth",
                 gamma=0.5
        ):
        super(CIDNet_SSM, self).__init__()

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
        
        # if cidnet_model_path != None:
        #     print(f"Loading CIDNet model from: {cidnet_model_path}")

        #     # Hugging Face Hub에서 CIDNet model 다운로드
        #     model_file = hf_hub_download(
        #         repo_id=cidnet_model_path,
        #         filename="model.safetensors",
        #         repo_type="model"
        #     )
        #     print(f"CIDNet model downloaded from: {model_file}")
            
        #     # 모델 초기화 및 가중치 로드
        #     state_dict = sf.load_file(model_file)
        #     self.load_state_dict(state_dict, strict=False)

        # if sam_model_path != None:
        #     self.sam = load_sam_model(sam_model_path)

        # Freeze all loaded parameters (before adding new layers)
        # for param in self.parameters():
        #     param.requires_grad = False
        

        self.alpha_predictor = SSM(in_channels=ch2*2, gamma=gamma)


    def forward(self, x, alpha_predict=True, base_alpha_s=1.0, base_alpha_i=1.0):
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

        # I_base: SMM 없이 기본 alpha 값만으로 변환한 결과 (Intermediate Supervision용)
        output_rgb_base = self.trans.HVI_to_RGB(output_hvi, base_alpha_s, base_alpha_i)

        if alpha_predict:
            alpha_input = torch.cat([i_dec1, hv_1], dim=1)  # (batch, ch1*2, h, w)
            alpha_s, alpha_i = self.alpha_predictor(alpha_input, base_alpha_s, base_alpha_i)
            output_rgb = self.trans.HVI_to_RGB(output_hvi, alpha_s, alpha_i)
        else:
            output_rgb = output_rgb_base

        # CIDNet_SSM은 항상 (output_rgb, output_rgb_base) 튜플 반환
        return output_rgb, output_rgb_base
    
    def RGB_to_HVI(self,x):
        hvi = self.trans.RGB_to_HVI(x)
        return hvi
    

if __name__ == "__main__":
    model = CIDNet_SSM()
    model.eval()
    dummy_input = torch.randn(2, 3, 400, 400)  # Example input tensor
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Should be (2, 3, 400, 400) for RGB output
    print("Model loaded and tested successfully.")