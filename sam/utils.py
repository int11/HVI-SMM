import os
import sys
import torch
import torchvision.transforms as transforms
from datetime import datetime
import dist
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from huggingface_hub import hf_hub_download
from net.CIDNet_SSM import CIDNet


class Tee:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, 'a')
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self  # Redirect stdout to this instance
        print(f"===== Logging session started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        return self

    def write(self, obj):
        self.file.write(obj)
        self.file.flush()
        self.stdout.write(obj)
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"===== Logging session ended {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        sys.stdout = self.stdout  # Restore original stdout
        self.file.close()


def checkpoint(epoch, model, optimizer, path):
    os.makedirs(path, exist_ok=True)
    model_out_path = os.path.join(path, f"epoch_{epoch}.pth")

    # Save model and optimizer states with epoch info
    # Use de_parallel to handle distributed models
    model_state = dist.de_parallel(model).state_dict() if dist.is_dist_available_and_initialized() else model.state_dict()

    checkpoint_dict = {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint_dict, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def extract_alpha_maps(model, image, device):
    """CIDNet_sam 모델에서 AlphaPredictor가 예측한 alpha_s, alpha_i 맵을 추출"""
    input_tensor = transforms.ToTensor()(image)
    
    # Padding
    factor = 8
    h, w = input_tensor.shape[1], input_tensor.shape[2]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input_tensor = torch.nn.functional.pad(input_tensor.unsqueeze(0), (0, padw, 0, padh), 'reflect')
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        # Forward pass through the model to get features
        dtypes = input_tensor.dtype
        hvi = model.trans.RGB_to_HVI(input_tensor)
        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)
        
        # low
        i_enc0 = model.IE_block0(i)
        i_enc1 = model.IE_block1(i_enc0)
        hv_0 = model.HVE_block0(hvi)
        hv_1 = model.HVE_block1(hv_0)
        i_jump0 = i_enc0
        hv_jump0 = hv_0
        
        i_enc2 = model.I_LCA1(i_enc1, hv_1)
        hv_2 = model.HV_LCA1(hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = model.IE_block2(i_enc2)
        hv_2 = model.HVE_block2(hv_2)
        
        i_enc3 = model.I_LCA2(i_enc2, hv_2)
        hv_3 = model.HV_LCA2(hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = model.IE_block3(i_enc2)
        hv_3 = model.HVE_block3(hv_2)
        
        i_enc4 = model.I_LCA3(i_enc3, hv_3)
        hv_4 = model.HV_LCA3(hv_3, i_enc3)
        
        i_dec4 = model.I_LCA4(i_enc4,hv_4)
        hv_4 = model.HV_LCA4(hv_4, i_enc4)
        
        hv_3 = model.HVD_block3(hv_4, hv_jump2)
        i_dec3 = model.ID_block3(i_dec4, v_jump2)
        i_dec2 = model.I_LCA5(i_dec3, hv_3)
        hv_2 = model.HV_LCA5(hv_3, i_dec3)
        
        hv_2 = model.HVD_block2(hv_2, hv_jump1)
        i_dec2 = model.ID_block2(i_dec3, v_jump1)
        
        i_dec1 = model.I_LCA6(i_dec2, hv_2)
        hv_1 = model.HV_LCA6(hv_2, i_dec2)
        
        
        
        i_dec1 = model.ID_block1(i_dec1, i_jump0)
        i_dec0 = model.ID_block0(i_dec1)
        hv_1 = model.HVD_block1(hv_1, hv_jump0)
        hv_0 = model.HVD_block0(hv_1)
        
        # Extract alpha maps from AlphaPredictor
        alpha_input = torch.cat([i_dec1, hv_1], dim=1)
        alpha_s, alpha_i = model.alpha_predictor(alpha_input, base_alpha_s=1.0, base_alpha_i=1.0)
        
        # Remove padding
        alpha_s = alpha_s[:, :h, :w]
        alpha_i = alpha_i[:, :, :h, :w]
        
        # Convert to numpy
        alpha_s_np = alpha_s.squeeze(0).cpu().numpy()  # (h, w)
        alpha_i_np = alpha_i.squeeze(0).cpu().numpy()  # (3, h, w)
    
    return alpha_s_np, alpha_i_np


def load_sam_model(sam_model_path="Gourieff/ReActor/models/sams/sam_vit_b_01ec64.pth", device=None):
    """Hugging Face에서 SAM 모델을 다운로드하고 로드"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # sam_model_path를 repo_id와 filename으로 분리
    parts = sam_model_path.split('/')
    if len(parts) < 3:
        raise ValueError("SAM model path should be in format: repo_id/filename (e.g., Gourieff/ReActor/models/sams/sam_vit_b_01ec64.pth)")
    
    model_repo = '/'.join(parts[:2])  # "Gourieff/ReActor"
    model_filename = '/'.join(parts[2:])  # "models/sams/sam_vit_b_01ec64.pth"
    
    print(f"Loading SAM model from: {model_repo}/{model_filename}")
    
    # Hugging Face Hub에서 SAM checkpoint 다운로드
    checkpoint_path = hf_hub_download(
        repo_id=model_repo, 
        filename=model_filename, 
        repo_type="dataset"
    )
    print(f"SAM model downloaded from: {checkpoint_path}")
    
    # SAM 모델 초기화 및 로드
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam = sam.to(device)  # Move SAM model to GPU
    
    # Mask generator 생성
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=200,
    )
    return mask_generator


def load_cidnet_sam_model(model_path, device):
    """CIDNet_sam 모델을 로드 (AlphaPredictor 포함)"""
    print(f"Loading CIDNet_sam model from: {model_path}")
    
    # 로컬 체크포인트 파일 로드
    checkpoint = torch.load(model_path, map_location=device)
    
    # CIDNet_sam.py의 CIDNet 클래스로 모델 초기화
    model = CIDNet(
        channels=[36, 36, 72, 144],
        heads=[1, 2, 4, 8],
        norm=False,
        cidnet_model_path=None,  # 사전학습 가중치 로드 안함
        sam_model_path=None,
        max_scale_factor=1.2
    )
    
    # 가중치 로드
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    print(f"CIDNet_sam model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
    return model