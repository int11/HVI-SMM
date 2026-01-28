import os
import sys
import torch
import torchvision.transforms as transforms
from datetime import datetime
import scripts.dist as dist
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from huggingface_hub import hf_hub_download
from net.CIDNet_SMM import CIDNet_SMM
from fvcore.nn import flop_count, FlopCountAnalysis


def compute_model_complexity(model, input_size=(1, 3, 384, 384)):
    """
    Compute model FLOPs and parameters using fvcore FlopCountAnalysis
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
    
    Returns:
        flops_str: Number of FLOPs (string format)
        params_str: Number of parameters (string format)
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Format parameters
    if total_params >= 1e9:
        params_str = f"{total_params / 1e9:.2f}G"
    elif total_params >= 1e6:
        params_str = f"{total_params / 1e6:.2f}M"
    elif total_params >= 1e3:
        params_str = f"{total_params / 1e3:.2f}K"
    else:
        params_str = f"{total_params}"
    
    # Get model device
    device = next(model.parameters()).device
    model.eval()
    
    # Use fvcore FlopCountAnalysis for accurate FLOPs calculation
    input_tensor = torch.randn(input_size, device=device)
    
    with torch.no_grad():
        flops_anal = FlopCountAnalysis(model, input_tensor)
        total_flops = flops_anal.total()
    
    # Format FLOPs
    if total_flops >= 1e12:
        flops_str = f"{total_flops / 1e12:.2f}T"
    elif total_flops >= 1e9:
        flops_str = f"{total_flops / 1e9:.2f}G"
    elif total_flops >= 1e6:
        flops_str = f"{total_flops / 1e6:.2f}M"
    elif total_flops >= 1e3:
        flops_str = f"{total_flops / 1e3:.2f}K"
    else:
        flops_str = f"{total_flops}"

    return flops_str, params_str

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
    model = CIDNet_SMM(
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