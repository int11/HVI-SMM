import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings

import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import lpips

from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from scipy.stats import norm


import safetensors.torch as sf
from huggingface_hub import hf_hub_download

from net.CIDNet import CIDNet
from sam.measure import metrics_one, calculate_psnr, calculate_ssim

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_cidnet_model(model_path):
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
    model = model.to(device)  # Move model to GPU
    model.eval()
    return model


def process_image_with_cidnet(model, image, alpha_s, alpha_i):
    input_tensor = transforms.ToTensor()(image)
    # Convert parameters to tensor
    if isinstance(alpha_s, np.ndarray):
        alpha_s = torch.from_numpy(alpha_s).to(device).to(input_tensor.dtype)
    else:
        alpha_s = torch.tensor(alpha_s, dtype=input_tensor.dtype, device=device)
    
    if isinstance(alpha_i, np.ndarray):
        alpha_i = torch.from_numpy(alpha_i).to(device).to(input_tensor.dtype)
    else:
        alpha_i = torch.tensor(alpha_i, dtype=input_tensor.dtype, device=device)
        
    factor = 8
    h, w = input_tensor.shape[1], input_tensor.shape[2]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input_tensor = torch.nn.functional.pad(input_tensor.unsqueeze(0), (0,padw,0,padh), 'reflect')
    input_tensor = input_tensor.to(device)  # Move input tensor to GPU
    
    with torch.no_grad():
        model.trans.gated = True
        model.trans.alpha_s = alpha_s
        model.trans.gated2 = True
        model.trans.alpha_i = alpha_i
        output = model(input_tensor)
    
    output = torch.clamp(output, 0, 1)
    output = output[:, :, :h, :w]
    output = output.cpu()  # Move output back to CPU for PIL conversion
    enhanced_img = transforms.ToPILImage()(output.squeeze(0))
    return enhanced_img


def determine_parameters(input_image, mask, gt_image, model, device, n_iterations=10):
    """베이지안 최적화를 사용하여 마스크 영역에 대한 최적의 파라미터를 찾는 함수"""
    
    # 파라미터 범위 설정 (alpha_s: 1.0~1.6, alpha_i: 0.8~1.2)
    param_bounds = {
        'alpha_s': (1.0, 1.6),
        'alpha_i': (0.8, 1.2)
    }

    # 초기 샘플 생성 (기본값과 그 근처 값들)
    X = np.array([
        [1.30, 1.00],
        [1.20, 0.98],
        [1.40, 1.02],
        [1.10, 0.96],
        [1.50, 1.04],
        [1.60, 0.90],
        [1.00, 1.10],
        [1.15, 0.85],
        [1.45, 1.15],
        [1.35, 0.80],
        [1.25, 1.20]
    ])
    
    # 목적 함수 계산 (마스크 영역의 이미지 품질 메트릭 기반)
    def objective_function(params):
        alpha_s, alpha_i = params
        
        # 현재 파라미터로 이미지 처리
        input_tensor = transforms.ToTensor()(input_image)
        factor = 8
        h, w = input_tensor.shape[1], input_tensor.shape[2]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_tensor = torch.nn.functional.pad(input_tensor.unsqueeze(0), (0, padw, 0, padh), 'reflect')
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            model.trans.alpha_s = alpha_s
            model.trans.alpha_i = alpha_i
            model.trans.gated = True
            model.trans.gated2 = True
            output = model(input_tensor)
            
        output = torch.clamp(output, 0, 1)
        output = output[:, :, :h, :w]
        output = output.cpu()
        enhanced_img = transforms.ToPILImage()(output.squeeze(0))
        
        # PIL Image를 numpy array로 변환
        enhanced_np = np.array(enhanced_img)
        input_np = np.array(input_image)
        gt_np = np.array(gt_image)
        
        # 마스크 적용
        mask_3d = np.stack([mask] * 3, axis=-1)
        enhanced_masked = np.where(mask_3d, enhanced_np, input_np)
        gt_masked = np.where(mask_3d, gt_np, input_np)
        
        # 마스크 영역에 대한 메트릭 계산
        # psnr_val = peak_signal_noise_ratio(enhanced_masked, gt_masked, data_range=255)
        # ssim_val = structural_similarity(enhanced_masked, gt_masked, channel_axis=2, data_range=255)
        psnr_val, ssim_val, lpips_val  = metrics_one(enhanced_masked, gt_masked, use_GT_mean=args.use_GT_mean, loss_fn=loss_fn)

        # 파라미터가 적절한 범위에 있는지 확인
        param_score = 1.0
        if not (param_bounds['alpha_s'][0] <= alpha_s <= param_bounds['alpha_s'][1] and
                param_bounds['alpha_i'][0] <= alpha_i <= param_bounds['alpha_i'][1]):
            param_score = 0.0
        
        # 최종 점수 계산 (PSNR, SSIM, 파라미터 범위 고려)
        score = (psnr_val / 40.0 + ssim_val + param_score)  
        return score

    # 최적화 전(기본값) 점수 계산
    default_score = objective_function([1.3, 1.0])

    y = np.array([objective_function(params) for params in X])

    # === Normalization (scaling) === 
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

    length_scale = 0.02
    kernel = Matern(nu=2.5, length_scale=length_scale, length_scale_bounds=(1e-5, 1e5))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=False, n_restarts_optimizer=10)

    for i in range(n_iterations):
        gp.fit(X_scaled, y_scaled)
        best_score = np.max(y_scaled)
        
        # 다음 파라미터 후보 생성 (원본 스케일)
        x_test = np.random.uniform(
            low=[param_bounds['alpha_s'][0], param_bounds['alpha_i'][0]],
            high=[param_bounds['alpha_s'][1], param_bounds['alpha_i'][1]],
            size=(100, 2)
        )
        # 후보를 스케일에 맞게 변환
        x_test_scaled = X_scaler.transform(x_test)
        
        # 예상 개선량 계산
        mean, std = gp.predict(x_test_scaled, return_std=True)
        ei = (mean - best_score) * norm.cdf((mean - best_score) / std) + std * norm.pdf((mean - best_score) / std)
        
        # 다음 파라미터 선택 (스케일 복원)
        next_params = x_test[np.argmax(ei)]
        next_score = objective_function(next_params)
        
        # 결과 업데이트
        X = np.vstack([X, next_params])
        y = np.append(y, next_score)
        X_scaled = X_scaler.fit_transform(X)
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
    
    # 최적의 파라미터 반환 (스케일 복원)
    best_idx = np.argmax(y)
    best_params = X[best_idx]

    return best_params[0], best_params[1], y[best_idx], default_score


def create_parameter_matrices(input_image, grouped_masks, gt_image, cidnet_model, device, n_iterations=10):
    """마스크 그룹들과 배경에 대해 최적의 파라미터를 찾아 매트릭스 생성"""
    
    # 모든 마스크를 합쳐서 배경 마스크 생성
    combined_mask = np.zeros_like(grouped_masks[0]['segmentation'], dtype=bool)
    for mask in grouped_masks:
        combined_mask |= mask['segmentation']
    background_mask = ~combined_mask
    
    # 배경 영역 처리
    bg_alpha_s, bg_alpha_i, best_score, default_score = determine_parameters(
        input_image, background_mask, gt_image, cidnet_model, device, n_iterations=n_iterations
    )
    print(f"Background parameters: alpha_s={bg_alpha_s:.4f}, alpha_i={bg_alpha_i:.4f}, Best score={best_score:.4f}, Default score={default_score:.4f}")

    # 파라미터 매트릭스 초기화
    alpha_s_matrix = np.zeros((input_image.height, input_image.width), dtype=float)
    alpha_i_matrix = np.zeros((input_image.height, input_image.width), dtype=float)

    # 배경 영역에 배경 파라미터 적용
    alpha_s_matrix[~combined_mask] = bg_alpha_s
    alpha_i_matrix[~combined_mask] = bg_alpha_i

    # 각 그룹 처리
    for i, mask in enumerate(grouped_masks):
        alpha_s, alpha_i, best_score, default_score = determine_parameters(
            input_image, mask['segmentation'], gt_image, cidnet_model, device, n_iterations=n_iterations
        )
        print(f"Group {i+1} parameters: alpha_s={alpha_s:.4f}, alpha_i={alpha_i:.4f}, Best score={best_score:.4f}, Default score={default_score:.4f}")

        # 기존 값이 있는 경우 평균 계산
        mask_indices = mask['segmentation']
        existing_values_s = alpha_s_matrix[mask_indices]
        existing_values_i = alpha_i_matrix[mask_indices]
        
        # 0이 아닌 값이 있는 경우에만 평균 계산
        non_zero_s = existing_values_s != 0
        non_zero_i = existing_values_i != 0
        
        alpha_s_matrix[mask_indices] = np.where(non_zero_s, 
            (existing_values_s + alpha_s) / 2, alpha_s)
        alpha_i_matrix[mask_indices] = np.where(non_zero_i,
            (existing_values_i + alpha_i) / 2, alpha_i)

    return alpha_s_matrix, alpha_i_matrix


def group_masks_by_stats(masks, num_groups=5):
    """마스크를 밝기와 대비를 기준으로 그룹화하고, 더 효과적인 클러스터링을 수행"""
    mask_stats = []
    
    for i, mask in enumerate(masks):
        mask_stats.append({
            'mask': mask,
            'area': mask['area']
        })
    
    area_values = np.array([m['area'] for m in mask_stats])
    
    # 정규화
    area_norm = (area_values - np.min(area_values)) / (np.max(area_values) - np.min(area_values))
    
    # 가중치 적용 (면적이 큰 마스크에 더 높은 가중치) 
    features = np.column_stack([area_norm])
    
    # 클러스터 개수는 최소(마스크 개수, num_groups)
    n_clusters = min(num_groups, len(features))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    grouped_masks = []
    for i in range(n_clusters):
        group_masks = [m['mask'] for m, label in zip(mask_stats, labels) if label == i]
        if not group_masks:
            continue
            
        combined_segmentation = np.zeros_like(group_masks[0]['segmentation'], dtype=bool)
        total_area = 0
        
        for mask in group_masks:
            combined_segmentation |= mask['segmentation']
            total_area += mask['area']
        
        grouped_mask = {
            'segmentation': combined_segmentation,
            'area': total_area
        }
        grouped_masks.append(grouped_mask)
    
    return grouped_masks


def visualize_mask_groups(image, grouped_masks):
    """마스크 그룹들을 다른 색깔로 표시한 이미지를 생성"""
    import matplotlib.pyplot as plt
    
    # 원본 이미지를 numpy array로 변환
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    # 색깔 팔레트 생성 (각 그룹별로 다른 색깔)
    colors = plt.cm.Set1(np.linspace(0, 1, len(grouped_masks)))
    
    # 마스크 시각화 이미지 생성
    mask_overlay = img_array.copy().astype(float)
    
    for i, mask_group in enumerate(grouped_masks):
        mask = mask_group['segmentation']
        color = colors[i][:3]  # RGB 값만 사용
        
        # 마스크 영역에 색깔 오버레이 (반투명 효과)
        for c in range(3):
            mask_overlay[mask, c] = mask_overlay[mask, c] * 0.7 + color[c] * 255 * 0.3
    
    mask_overlay = np.clip(mask_overlay, 0, 255).astype(np.uint8)
    
    # 결과 이미지들 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Mask Groups Visualization', fontsize=16)
    
    # 원본 이미지
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 마스크 오버레이
    axes[0, 1].imshow(mask_overlay)
    axes[0, 1].set_title(f'Mask Groups Overlay ({len(grouped_masks)} groups)')
    axes[0, 1].axis('off')
    
    # 개별 마스크 그룹들
    axes[1, 0].imshow(img_array)
    for i, mask_group in enumerate(grouped_masks):
        mask = mask_group['segmentation']
        color = colors[i][:3]
        
        # 마스크 경계선 그리기
        from scipy import ndimage
        mask_edges = ndimage.binary_dilation(mask) ^ mask
        for c in range(3):
            img_array[mask_edges, c] = color[c] * 255
    
    axes[1, 0].imshow(img_array)
    axes[1, 0].set_title('Mask Group Boundaries')
    axes[1, 0].axis('off')
    
    # 배경 마스크
    combined_mask = np.zeros_like(grouped_masks[0]['segmentation'], dtype=bool)
    for mask in grouped_masks:
        combined_mask |= mask['segmentation']
    background_mask = ~combined_mask
    
    background_vis = img_array.copy()
    # 배경 영역을 빨간색으로 칠하기
    background_vis[background_mask] = [255, 0, 0]  # 빨간색
    axes[1, 1].imshow(background_vis.astype(np.uint8))
    axes[1, 1].set_title('Background Area (red regions)')
    axes[1, 1].axis('off')
    
    # 범례 추가
    legend_elements = []
    for i, mask_group in enumerate(grouped_masks):
        area = mask_group['area']
        color = colors[i][:3]
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                        markerfacecolor=color, markersize=10,
                                        label=f'Group {i+1} (area: {area})'))
    
    axes[0, 1].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    
    # 시각화 창 표시
    print(f"Showing mask groups visualization (close window to continue)")
    print(f"  Total groups: {len(grouped_masks)}")
    for i, mask_group in enumerate(grouped_masks):
        print(f"  Group {i+1}: area = {mask_group['area']}")
    
    # 창 표시 (블로킹 방식)
    plt.show()
    
    # 메모리 정리는 자동으로 됨


def sort_files_by_number(file_list):
    """파일 리스트를 숫자 순서대로 정렬"""
    import re
    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    return sorted(file_list, key=extract_number)


def parse_args():
    parser = argparse.ArgumentParser(description='Process images using CIDNet and SAM')
    parser.add_argument('--dir', type=str, default="datasets/LOLdataset/our485",
                        help='Base directory containing low/high subdirectories and for saving matrices')
    parser.add_argument('--output_dir', type=str, default="results/LOLdataset",
                        help='Directory to save output images')
    parser.add_argument('--cidnet_model', type=str, default="Fediory/HVI-CIDNet-LOLv1-woperc",
                        help='CIDNet model name or path from Hugging Face')
    parser.add_argument('--sam_model', type=str, default="Gourieff/ReActor/models/sams/sam_vit_b_01ec64.pth",
                        help='SAM model path in format: repo_id/filename (e.g., Gourieff/ReActor/models/sams/sam_vit_b_01ec64.pth)')
    parser.add_argument('--visualize_masks', action='store_true', default=False,
                        help='Save mask group visualization images')
    parser.add_argument('--num_groups', type=int, default=10,
                        help='Number of mask groups to create')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations for Bayesian optimization')
    parser.add_argument('--use_GT_mean', action='store_true', default=True,
                        help='Use the mean of GT to rectify the output of the model')
    return parser.parse_args()


if __name__ == "__main__":
    from train import Tee
    with Tee(os.path.join(f'./sam/log.txt')):
        args = parse_args()
        
        # Add device detection
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Set up directories based on --dir argument
        input_dir = os.path.join(args.dir, "low")
        gt_dir = os.path.join(args.dir, "high")
        matrix_dir_base = args.dir
        
        # Create subdirectories for different outputs
        comparison_dir = os.path.join(args.output_dir, "comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Get list of PNG files from input directory and sort by number
        input_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
        input_files = sort_files_by_number(input_files)
        
        # Load models only once
        cidnet_model = load_cidnet_model(args.cidnet_model)
        sam_model = load_sam_model(args.sam_model)
        # Calculate metrics for both whole and SAM enhanced images
        loss_fn = lpips.LPIPS(net='alex')
        loss_fn.cuda()
        
        # Process each image
        whole_metrics = {'psnr': [], 'ssim': [], 'lpips': []}
        sam_metrics = {'psnr': [], 'ssim': [], 'lpips': []}
        
        for input_file in input_files:
            print(f"Processing {input_file}...")
            input_filename = os.path.splitext(os.path.basename(input_file))[0]
            
            # Load input and ground truth images
            input_path = os.path.join(input_dir, input_file)
            gt_path = os.path.join(gt_dir, input_file)
            
            input_image = Image.open(input_path).convert('RGB')
            gt_image = Image.open(gt_path).convert('RGB')
            
            # 1. 통으로 CIDNet 처리
            whole_enhanced = process_image_with_cidnet(cidnet_model, input_image, 1.3, 1.0)
            whole_psnr, whole_ssim, whole_lpips = metrics_one(whole_enhanced, gt_image, use_GT_mean=args.use_GT_mean, loss_fn=loss_fn)
            whole_metrics['psnr'].append(whole_psnr)
            whole_metrics['ssim'].append(whole_ssim)
            whole_metrics['lpips'].append(whole_lpips.item())

            # 1.5. 기본 파라미터로 통으로 CIDNet 처리
            default_enhanced = process_image_with_cidnet(cidnet_model, input_image, 1.0, 1.0)
            default_psnr, default_ssim, default_lpips = metrics_one(default_enhanced, gt_image, use_GT_mean=args.use_GT_mean, loss_fn=loss_fn)


            # 2. 영역별 CIDNet 처리
            initial_masks = sam_model.generate(np.array(input_image))

            # 마스크 그룹핑
            grouped_masks = group_masks_by_stats(
                initial_masks, num_groups=args.num_groups
            )

            if args.visualize_masks:
                visualize_mask_groups(input_image, grouped_masks)

            # 파라미터 매트릭스 생성
            alpha_s_matrix, alpha_i_matrix = create_parameter_matrices(
                input_image, grouped_masks, gt_image, cidnet_model, device, n_iterations=args.iterations
            )
            matrix_dir = f"{matrix_dir_base}/iter{args.iterations}_num_groups{args.num_groups}"
            os.makedirs(matrix_dir, exist_ok=True)
            torch.save({
                'alpha_s': alpha_s_matrix,
                'alpha_i': alpha_i_matrix
            }, os.path.join(matrix_dir, f"{input_filename}.pth"))

            output_image = process_image_with_cidnet(cidnet_model, input_image, alpha_s_matrix, alpha_i_matrix)
            
            # Calculate metrics for SAM enhanced image
            sam_psnr, sam_ssim, sam_lpips = metrics_one(output_image, gt_image, use_GT_mean=args.use_GT_mean, loss_fn=loss_fn)
            sam_metrics['psnr'].append(sam_psnr)
            sam_metrics['ssim'].append(sam_ssim)
            sam_metrics['lpips'].append(sam_lpips.item())
            
            # 3. 결과 비교를 위한 시각화
            print(f"  Whole(1.3,1.0) - PSNR: {whole_psnr:.4f}, SSIM: {whole_ssim:.4f}, LPIPS: {whole_lpips.item():.4f}"
                  f" | Default(1.0,1.0) - PSNR: {default_psnr:.4f}, SSIM: {default_ssim:.4f}, LPIPS: {default_lpips.item():.4f}"
                  f" | SAM Enhanced - PSNR: {sam_psnr:.4f}, SSIM: {sam_ssim:.4f}, LPIPS: {sam_lpips.item():.4f}")
            comparison = Image.new('RGB', (input_image.width * 4, input_image.height + 50))
            comparison.paste(input_image, (0, 0))
            comparison.paste(gt_image, (input_image.width, 0))
            comparison.paste(whole_enhanced, (input_image.width * 2, 0))
            comparison.paste(output_image, (input_image.width * 3, 0))
            draw = ImageDraw.Draw(comparison)
            font = ImageFont.load_default()
            draw.text((input_image.width//2 - 20, input_image.height + 10), "Input", fill="white", font=font)
            draw.text((input_image.width + input_image.width//2 - 15, input_image.height + 10), "GT", fill="white", font=font)
            draw.text((input_image.width*2 + input_image.width//2 - 25, input_image.height + 10), "Whole", fill="white", font=font)
            draw.text((input_image.width*3 + input_image.width//2 - 20, input_image.height + 10), "SAM", fill="white", font=font)
            
            comparison_path = os.path.join(comparison_dir, f"{input_filename}.png")
            comparison.save(comparison_path)
        
        # Calculate and print average metrics
        print("\n=== Reimplementation of paper SOTA ===")
        print(f"Average PSNR: {np.mean(whole_metrics['psnr']):.4f} dB")
        print(f"Average SSIM: {np.mean(whole_metrics['ssim']):.4f}")
        print(f"Average LPIPS: {np.mean(whole_metrics['lpips']):.4f}")
        
        print("\n=== SAM Enhanced Image Metrics ===")
        print(f"Average PSNR: {np.mean(sam_metrics['psnr']):.4f} dB")
        print(f"Average SSIM: {np.mean(sam_metrics['ssim']):.4f}")
        print(f"Average LPIPS: {np.mean(sam_metrics['lpips']):.4f}")