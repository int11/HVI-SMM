import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data import *
from loss.losses import *
from scripts.measure import metrics
import scripts.dist as dist
from scripts.options import option, load_datasets
import safetensors.torch as sf
from huggingface_hub import hf_hub_download
from net.CIDNet_SMM import CIDNet_SMM
from net.CIDNet import CIDNet
from net.CIDNet_fix import CIDNet_fix
from net.BaseCIDNet import BaseCIDNet
from net.BaseCIDNetWithSMM import BaseCIDNet_SMM
from torchvision.transforms import ToPILImage, ToTensor
import torch.utils.data
import matplotlib.pyplot as plt


def array_to_heatmap(arr, cmap='jet'):
    """Convert numpy array to PIL heatmap image"""
    # 데이터의 1~99 분위수로 vmin/vmax 자동 설정
    vmin = np.percentile(arr, 1)
    vmax = np.percentile(arr, 99)
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6  # 0으로 나누기 방지
    normalized = (arr - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)
    colored = plt.cm.get_cmap(cmap)(normalized)
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode='RGB')


def eval(model, data, alpha_combinations):
    """
    Unified evaluation function for both CIDNet and CIDNet_SMM models.
    Efficiently handles multiple alpha combinations by computing features once,
    then applying different alpha values without redundant forward passes.
    
    Args:
        model: BaseCIDNet or BaseCIDNet_SMM model instance
        data: DataLoader for test data
        alpha_combinations: List of (base_alpha_s, base_alpha_i, alpha_rgb) tuples
                           예: [(1.0, 1.0, 1.0)] or [(1.3, 1.0, 1.0), (1.0, 1.0, 0.8)]
        
    Returns:
        dict: {alpha_combo: (output_list, gt_list)}
              예: {(1.0, 1.0, 1.0): ([output_np, ...], [gt_pil, ...])}
    
    Note:
        Uses forward_features() and apply_alpha_scaling() to avoid redundant
        computations across different alpha values. Works for both base CIDNet
        and SMM-enhanced models.
    """
    if not isinstance(model, BaseCIDNet):
        raise ValueError("eval() only supports BaseCIDNet and its subclasses")
    
    model = dist.de_parallel(model)
    model.eval()
    torch.set_grad_enabled(False)
    device = dist.get_device()
    
    # Initialize results dictionary
    results = {combo: ([], []) for combo in alpha_combinations}
    
    for batch in data:
        input_img = batch[0].to(device)
        gt_img = batch[1]
        
        # Forward pass once (heavy computation)
        with torch.no_grad():
            output_hvi, scale_factor = model.forward_features(input_img)
        
        # Apply different alpha combinations (lightweight)
        for (base_alpha_s, base_alpha_i, alpha_rgb) in alpha_combinations:
            with torch.no_grad():
                output_rgb = model.apply_alpha_scaling(
                    output_hvi, scale_factor,
                    base_alpha_s, base_alpha_i, alpha_rgb
                )
            
            # Convert to numpy and store
            output_rgb = torch.clamp(output_rgb, 0, 1)
            output_np = output_rgb.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            gt_pil = ToPILImage()(gt_img.squeeze(0).cpu())
            
            results[(base_alpha_s, base_alpha_i, alpha_rgb)][0].append(output_np)
            results[(base_alpha_s, base_alpha_i, alpha_rgb)][1].append(gt_pil)
        
        torch.cuda.empty_cache()
    
    torch.set_grad_enabled(True)
    return results


def load_cidnet_base_model(model_path, device):
    """Hugging Face에서 CIDNet 모델을 다운로드하고 로드"""
    print(f"Loading CIDNet model from: {model_path}")
    
    # Hugging Face Hub에서 CIDNet model 다운로드
    model_file = hf_hub_download(
        repo_id=model_path, 
        filename="model.safetensors", 
        repo_type="model"
    )
    print(f"CIDNet model downloaded from: {model_file}")
    
    # 모델 초기화 및 가중치 로드
    model = CIDNet()
    state_dict = sf.load_file(model_file)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model

    
if __name__ == '__main__':
    parser = option()
    parser.add_argument('--input_image', type=str, default=None, help='Path to input image')
    parser.add_argument('--cidnet_ssm_path', type=str, default="weights/lolv2_real/v5/20260112_162957_gt_mean_loss='hvi'/epoch_570.pth", help='Path to the CIDNet SSM model weights')
    parser.add_argument('--output_dir', type=str, default='results/ssm_eval_results_lolv2realv2_2', help='Directory to save comparison images')
        # Available CIDNet models from Hugging Face:
        # Fediory/HVI-CIDNet
        # Fediory/HVI-CIDNet-LOLv1-wperc
        # Fediory/HVI-CIDNet-LOLv1-woperc
        # Fediory/HVI-CIDNet-LOLv2-real-bestPSNR
        # Fediory/HVI-CIDNet-LOLv2-real-bestSSIM
        # Fediory/HVI-CIDNet-LOLv2-syn-wperc
        # Fediory/HVI-CIDNet-LOLv2-syn-woperc
        # Fediory/HVI-CIDNet-Generalization
        # Fediory/HVI-CIDNet-LOL-Blur
        # Fediory/HVI-CIDNet-SICE
        # Fediory/HVI-CIDNet-Sony-Total-Dark
        # Fediory/HVI-CIDNet-FiveK
    parser.add_argument('--cidnet_path', type=str, default="Fediory/HVI-CIDNet-LOLv2-real-bestPSNR",
                        help='Path to the base CIDNet model')
    parser.add_argument('--base_alpha_s', type=float, default=1.0, help='Base alpha_s parameter for CIDNet')
    parser.add_argument('--base_alpha_i', type=float, default=1.0, help='Base alpha_i parameter for CIDNet')
    parser.add_argument('--alpha_rgb', type=float, default=0.8, help='RGB scaling factor')
    parser.add_argument('--use_GT_mean', type=bool, default=False, help='Use the mean of GT to rectify the output of the model')
    args = parser.parse_args()
    
    # Create alpha_combinations from parsed arguments
    alpha_combinations = [(args.base_alpha_s, args.base_alpha_i, args.alpha_rgb)]

    if args.input_image is not None:
        # 단일 이미지 평가 모드
        from PIL import Image
        import numpy as np
        
        input_img = Image.open(args.input_image)
        
        # Convert to DataLoader format
        input_tensor = ToTensor()(input_img).unsqueeze(0)
        gt_tensor = torch.zeros_like(input_tensor)  # Dummy GT for single image
        data_loader = [(input_tensor, gt_tensor)]
        
    else:
        # 데이터셋 평가 모드
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        training_data_loader, testing_data_loader = load_datasets(args)
        data_loader = testing_data_loader
    
    # Load CIDNet_SSM model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_net = CIDNet_SMM().cuda()
    checkpoint_data = torch.load(args.cidnet_ssm_path, map_location=lambda storage, loc: storage)
    eval_net.load_state_dict(checkpoint_data['model_state_dict'])
    print(f"Loaded CIDNet_SSM checkpoint from {args.cidnet_ssm_path}")
    
    # Hook to capture alpha maps from CIDNet_SSM
    alpha_outputs = []
    def hook_fn(module, input, output):
        alpha_outputs.append(output)
    hook = eval_net.alpha_predictor.register_forward_hook(hook_fn)

    # Evaluate - CIDNet_SSM with alpha prediction
    results = eval(eval_net, data_loader, alpha_combinations)
    output_list, gt_list = results[alpha_combinations[0]]
    
    # Remove hook
    hook.remove()
    
    if args.input_image is not None:
        # 단일 이미지 저장
        os.makedirs(args.output_dir, exist_ok=True)
        output_img = Image.fromarray((output_list[0] * 255).astype(np.uint8))
        output_path = os.path.join(args.output_dir, 'single_image_output.png')
        output_img.save(output_path)
        print(f"Saved single image output to {output_path}")
        
        # Save alpha maps if available
        if alpha_outputs:
            scale_factor = alpha_outputs[0]  # shape: (1, 2, h, w)
            alpha_s_np = scale_factor[0, 0, :, :].cpu().numpy()
            alpha_i_np = scale_factor[0, 1, :, :].cpu().numpy()
            alpha_s_img = array_to_heatmap(alpha_s_np)
            alpha_i_img = array_to_heatmap(alpha_i_np)
            
            alpha_s_img.save(os.path.join(args.output_dir, 'single_image_alpha_s.png'))
            alpha_i_img.save(os.path.join(args.output_dir, 'single_image_alpha_i.png'))
            print(f"Saved alpha maps to {args.output_dir}")
        
    else:
        # 데이터셋 모드 - 비교 이미지 생성
        # Load base CIDNet model
        cidnet_base = load_cidnet_base_model(args.cidnet_path, device)
        print(f"Loaded base CIDNet model from {args.cidnet_path}")

        # Extract alpha maps for dataset
        # alpha_outputs contains scale_factor tensors with shape (batch_size, 2, h, w)
        alpha_s_list = []
        alpha_i_list = []
        for out in alpha_outputs:
            # out shape: (batch_size, 2, h, w)
            for b in range(out.shape[0]):
                alpha_s_list.append(out[b, 0, :, :].cpu().numpy())
                alpha_i_list.append(out[b, 1, :, :].cpu().numpy())

        # Evaluate - Base CIDNet with proper alpha combinations
        results_base = eval(cidnet_base, data_loader, alpha_combinations)
        output_base_list, gt_list_base = results_base[alpha_combinations[0]]
    
        # Calculate metrics for CIDNet_sam
        print("\n" + "="*60)
        print("CIDNet_SSM with Alpha Prediction")
        print("="*60)
        avg_psnr_sam, avg_ssim_sam, avg_lpips_sam = metrics(output_list, gt_list, use_GT_mean=args.use_GT_mean)
        print(f"PSNR: {avg_psnr_sam:.4f} dB || SSIM: {avg_ssim_sam:.4f} || LPIPS: {avg_lpips_sam:.4f}")
        
        # Calculate metrics for Base CIDNet
        print("\n" + "="*60)
        print("Base CIDNet (Standard Parameters)")
        print("="*60)
        avg_psnr_base, avg_ssim_base, avg_lpips_base = metrics(output_base_list, gt_list, use_GT_mean=args.use_GT_mean)
        print(f"PSNR: {avg_psnr_base:.4f} dB || SSIM: {avg_ssim_base:.4f} || LPIPS: {avg_lpips_base:.4f}")
        
        # Print comparison
        print("\n" + "="*60)
        print("Performance Comparison")
        print("="*60)
        print(f"PSNR Improvement: {avg_psnr_sam - avg_psnr_base:+.4f} dB")
        print(f"SSIM Improvement: {avg_ssim_sam - avg_ssim_base:+.4f}")
        print(f"LPIPS Improvement: {avg_lpips_sam - avg_lpips_base:+.4f}")
        
        # Save comparison images
        os.makedirs(args.output_dir, exist_ok=True)
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # Get input images
        input_images = [batch[0].squeeze(0).numpy().transpose(1, 2, 0) for batch in testing_data_loader]
        
        # Define directories
        dirs = ['input', 'cidnet', 'cidnet_ssm', 'gt', 'alpha_s', 'alpha_i', 'comparison']
        dir_paths = {name: os.path.join(args.output_dir, name) for name in dirs}
        
        # Create directories
        for path in dir_paths.values():
            os.makedirs(path, exist_ok=True)
        
        for idx, (output_np, base_output_np, gt_img, input_np, alpha_s_np, alpha_i_np) in enumerate(zip(output_list, output_base_list, gt_list, input_images, alpha_s_list, alpha_i_list)):
            # Convert to PIL images
            images = {
                'input': Image.fromarray((input_np * 255).astype(np.uint8)),
                'cidnet': Image.fromarray((base_output_np * 255).astype(np.uint8)),
                'cidnet_ssm': Image.fromarray((output_np * 255).astype(np.uint8)),
                'gt': gt_img,
                'alpha_s': array_to_heatmap(alpha_s_np),
                'alpha_i': array_to_heatmap(alpha_i_np)
            }
        
            
            # Save individual images
            img_name = f'{idx+1:03d}.png'
            for name, img in images.items():
                if name != 'comparison':
                    img.save(os.path.join(dir_paths[name], img_name))
            
            # Create and save comparison
            h, w = output_np.shape[:2]
            comparison = Image.new('RGB', (w * 6, h + 40))
            comparison.paste(images['input'], (0, 40))
            comparison.paste(images['cidnet'], (w, 40))
            comparison.paste(images['cidnet_ssm'], (w * 2, 40))
            comparison.paste(images['gt'], (w * 3, 40))
            comparison.paste(images['alpha_s'], (w * 4, 40))
            comparison.paste(images['alpha_i'], (w * 5, 40))
            
            draw = ImageDraw.Draw(comparison)
            font = ImageFont.load_default()
            labels = ["Input", "Base CIDNet", "CIDNet_SSM", "GT", "Alpha_s", "Alpha_i"]
            for i, label in enumerate(labels):
                draw.text((w * i + w//2 - len(label)*3, 10), label, fill="white", font=font)
            
            comparison.save(os.path.join(dir_paths['comparison'], f'comparison_{idx+1:03d}.png'))
            print(f"Saved images [{idx+1:03d}]: {', '.join(images.keys())}")
        
        print(f"\n✓ Saved {len(output_list)} images to:")
        for name, path in dir_paths.items():
            print(f"  - {name.capitalize()}: {path}")