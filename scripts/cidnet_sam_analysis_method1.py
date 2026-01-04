import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage

from sam_metrics import group_masks_by_stats, sort_files_by_number
from utils import extract_alpha_maps, load_sam_model, load_cidnet_sam_model
from data.options import option, load_datasets

warnings.filterwarnings("ignore")


def calculate_group_alpha_stats(alpha_map, grouped_masks):
    """각 SAM 그룹 내에서 alpha 값의 통계 계산"""
    stats = []
    
    # Background 통계
    combined_mask = np.zeros_like(grouped_masks[0]['segmentation'], dtype=bool)
    for mask in grouped_masks:
        combined_mask |= mask['segmentation']
    background_mask = ~combined_mask
    
    if np.any(background_mask):
        bg_values = alpha_map[background_mask]
        stats.append({
            'group_id': 'Background',
            'mean': np.mean(bg_values),
            'std': np.std(bg_values),
            'min': np.min(bg_values),
            'max': np.max(bg_values),
            'area': np.sum(background_mask)
        })
    
    # 각 그룹별 통계
    for i, mask_group in enumerate(grouped_masks):
        mask = mask_group['segmentation']
        values = alpha_map[mask]
        stats.append({
            'group_id': i + 1,
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'area': mask_group['area']
        })
    
    return stats


def visualize_alpha_with_sam_masks(image, alpha_s, alpha_i, grouped_masks, output_path=None, output_dir=None, filename_prefix=None):
    """
    1번 방법: Alpha 맵 시각화 및 SAM 마스크 오버레이
    - Alpha_s, Alpha_i를 히트맵으로 시각화
    - SAM 마스크 경계선을 함께 표시
    - 같은 SAM 그룹 내에서 alpha 값들의 통계 출력
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    # Alpha_i는 RGB 채널별이므로 평균을 취함
    alpha_i_mean = np.mean(alpha_i, axis=0)
    
    # Alpha_s와 Alpha_i를 개별 PNG 파일로 저장 (jet colormap 사용)
    if output_dir and filename_prefix:
        # Alpha_s를 jet colormap으로 변환하여 저장
        alpha_s_normalized = (alpha_s - alpha_s.min()) / (alpha_s.max() - alpha_s.min())
        alpha_s_colored = plt.cm.jet(alpha_s_normalized)  # RGBA (0-1 범위)
        alpha_s_rgb = (alpha_s_colored[:, :, :3] * 255).astype(np.uint8)  # RGB로 변환
        # alpha_s 폴더에 저장
        alpha_s_folder = os.path.join(output_dir, "alpha_s")
        os.makedirs(alpha_s_folder, exist_ok=True)
        alpha_s_path = os.path.join(alpha_s_folder, f"{filename_prefix}.png")
        Image.fromarray(alpha_s_rgb, mode='RGB').save(alpha_s_path)
        print(f"  Saved alpha_s map to: {alpha_s_path}")
        
        # Alpha_i를 jet colormap으로 변환하여 저장
        alpha_i_normalized = (alpha_i_mean - alpha_i_mean.min()) / (alpha_i_mean.max() - alpha_i_mean.min())
        alpha_i_colored = plt.cm.jet(alpha_i_normalized)  # RGBA (0-1 범위)
        alpha_i_rgb = (alpha_i_colored[:, :, :3] * 255).astype(np.uint8)  # RGB로 변환
        # alpha_i 폴더에 저장
        alpha_i_folder = os.path.join(output_dir, "alpha_i")
        os.makedirs(alpha_i_folder, exist_ok=True)
        alpha_i_path = os.path.join(alpha_i_folder, f"{filename_prefix}.png")
        Image.fromarray(alpha_i_rgb, mode='RGB').save(alpha_i_path)
        print(f"  Saved alpha_i map to: {alpha_i_path}")
    
    # SAM 마스크 경계선 생성
    mask_boundaries = np.zeros_like(img_array)
    colors_list = plt.cm.Set1(np.linspace(0, 1, len(grouped_masks)))
    
    for i, mask_group in enumerate(grouped_masks):
        mask = mask_group['segmentation']
        color = colors_list[i][:3]
        
        # 경계선 추출
        mask_edges = ndimage.binary_dilation(mask) ^ mask
        for c in range(3):
            mask_boundaries[mask_edges, c] = color[c] * 255
    
    # Figure 생성
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle('Alpha Maps vs SAM Mask Groups', fontsize=16, fontweight='bold')
    
    # 1. 원본 이미지
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')
    
    # 2. SAM 마스크 그룹
    axes[0, 1].imshow(img_array)
    axes[0, 1].imshow(mask_boundaries, alpha=0.8)
    axes[0, 1].set_title(f'SAM Mask Groups ({len(grouped_masks)} groups)', fontsize=12)
    axes[0, 1].axis('off')
    
    # 3. Alpha_s 히트맵
    im1 = axes[0, 2].imshow(alpha_s, cmap='jet', vmin=alpha_s.min(), vmax=alpha_s.max())
    axes[0, 2].set_title(f'Alpha_s Map (range: {alpha_s.min():.3f}-{alpha_s.max():.3f})', fontsize=12)
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # 4. Alpha_s + SAM 경계선
    im2 = axes[1, 0].imshow(alpha_s, cmap='jet', vmin=alpha_s.min(), vmax=alpha_s.max())
    # SAM 경계선을 흰색으로 오버레이
    for i, mask_group in enumerate(grouped_masks):
        mask = mask_group['segmentation']
        mask_edges = ndimage.binary_dilation(mask) ^ mask
        alpha_s_overlay = np.ma.masked_where(~mask_edges, np.ones_like(alpha_s))
        axes[1, 0].imshow(alpha_s_overlay, cmap='gray_r', alpha=0.8, vmin=0, vmax=1)
    axes[1, 0].set_title('Alpha_s + SAM Boundaries', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # 5. Alpha_i 히트맵 (평균)
    im3 = axes[1, 1].imshow(alpha_i_mean, cmap='jet', vmin=alpha_i_mean.min(), vmax=alpha_i_mean.max())
    axes[1, 1].set_title(f'Alpha_i Map (mean, range: {alpha_i_mean.min():.3f}-{alpha_i_mean.max():.3f})', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # 6. Alpha_i + SAM 경계선
    im4 = axes[1, 2].imshow(alpha_i_mean, cmap='jet', vmin=alpha_i_mean.min(), vmax=alpha_i_mean.max())
    for i, mask_group in enumerate(grouped_masks):
        mask = mask_group['segmentation']
        mask_edges = ndimage.binary_dilation(mask) ^ mask
        alpha_i_overlay = np.ma.masked_where(~mask_edges, np.ones_like(alpha_i_mean))
        axes[1, 2].imshow(alpha_i_overlay, cmap='gray_r', alpha=0.8, vmin=0, vmax=1)
    axes[1, 2].set_title('Alpha_i + SAM Boundaries', fontsize=12)
    axes[1, 2].axis('off')
    plt.colorbar(im4, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # 통계 계산 및 출력
    print("\n" + "="*80)
    print("Alpha Statistics by SAM Groups")
    print("="*80)
    
    alpha_s_stats = calculate_group_alpha_stats(alpha_s, grouped_masks)
    alpha_i_stats = calculate_group_alpha_stats(alpha_i_mean, grouped_masks)
    
    print(f"\n{'Group':<12} {'Area':<10} {'Alpha_s Mean':<15} {'Alpha_s Std':<15} {'Alpha_i Mean':<15} {'Alpha_i Std':<15}")
    print("-"*80)
    
    for s_stat, i_stat in zip(alpha_s_stats, alpha_i_stats):
        group_id = s_stat['group_id']
        area = s_stat['area']
        print(f"{str(group_id):<12} {area:<10} {s_stat['mean']:<15.4f} {s_stat['std']:<15.4f} {i_stat['mean']:<15.4f} {i_stat['std']:<15.4f}")
    
    print("="*80)
    
    # 그룹 내 분산 vs 그룹 간 평균 차이 분석
    s_means = [stat['mean'] for stat in alpha_s_stats]
    s_stds = [stat['std'] for stat in alpha_s_stats]
    i_means = [stat['mean'] for stat in alpha_i_stats]
    i_stds = [stat['std'] for stat in alpha_i_stats]
    
    print(f"\nWithin-group variation (avg std):")
    print(f"  Alpha_s: {np.mean(s_stds):.4f}")
    print(f"  Alpha_i: {np.mean(i_stds):.4f}")
    
    print(f"\nBetween-group variation (std of means):")
    print(f"  Alpha_s: {np.std(s_means):.4f}")
    print(f"  Alpha_i: {np.std(i_means):.4f}")
    
    print(f"\nVariation ratio (between/within):")
    print(f"  Alpha_s: {np.std(s_means) / np.mean(s_stds):.4f}")
    print(f"  Alpha_i: {np.std(i_means) / np.mean(i_stds):.4f}")
    print("\n(Higher ratio = alpha values are more consistent within groups)")
    print("="*80 + "\n")
    
    return alpha_s_stats, alpha_i_stats


def parse_args():
    parser = option()
    # Override default dataset for analysis script
    parser.set_defaults(dataset='lolv2_syn')
    parser.add_argument('--model_path', type=str, default="weights/lolv2_syn/20251104_153530_w_perc_msf1.2/epoch_500.pth",
                        help='Path to CIDNet_sam model checkpoint')
    parser.add_argument('--output_dir', type=str, default="results/analysis_results",
                        help='Directory to save analysis results')
    parser.add_argument('--sam_model', type=str, default="Gourieff/ReActor/models/sams/sam_vit_b_01ec64.pth",
                        help='SAM model path')
    parser.add_argument('--num_groups', type=int, default=10,
                        help='Number of mask groups to create')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process (None = all)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    method1_dir = os.path.join(args.output_dir, "method1_alpha_heatmap")
    os.makedirs(method1_dir, exist_ok=True)
    
    # 각 이미지 타입별 폴더 생성
    alpha_s_dir = os.path.join(method1_dir, "alpha_s")
    alpha_i_dir = os.path.join(method1_dir, "alpha_i")
    input_dir = os.path.join(method1_dir, "input_image")
    gt_dir = os.path.join(method1_dir, "gt_image")
    analysis_dir = os.path.join(method1_dir, "analysis")
    
    os.makedirs(alpha_s_dir, exist_ok=True)
    os.makedirs(alpha_i_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Load datasets using load_datasets function
    training_data_loader, testing_data_loader = load_datasets(args)
    
    # 모델 로드
    cidnet_sam_model = load_cidnet_sam_model(args.model_path, device)
    sam_model = load_sam_model(args.sam_model, device)
    
    # 각 이미지 처리
    image_count = 0
    for batch in testing_data_loader:
        if args.max_images and image_count >= args.max_images:
            break
            
        input_tensor, gt_tensor, name = batch[0], batch[1], batch[2]
        
        # Convert tensor to PIL Image
        from torchvision.transforms import ToPILImage
        input_image = ToPILImage()(input_tensor.squeeze(0).cpu())
        gt_image = ToPILImage()(gt_tensor.squeeze(0).cpu())
        
        # name 처리 (문자열 또는 튜플일 수 있음)
        if isinstance(name, (list, tuple)):
            input_filename = name[0]
        else:
            input_filename = name
        
        # 확장자 제거
        filename_without_ext = os.path.splitext(input_filename)[0]
        
        image_count += 1
        print(f"\n[{image_count}] Processing {input_filename}...")
        
        # Input image 저장
        input_image.save(os.path.join(input_dir, f"{image_count:03d}.png"))
        print(f"  Saved input image to: {os.path.join(input_dir, f'{image_count:03d}.png')}")
        
        # GT image 저장
        gt_image.save(os.path.join(gt_dir, f"{image_count:03d}.png"))
        print(f"  Saved gt image to: {os.path.join(gt_dir, f'{image_count:03d}.png')}")
        
        # SAM 마스크 생성 및 그룹핑
        print("  Generating SAM masks...")
        initial_masks = sam_model.generate(np.array(input_image))
        grouped_masks = group_masks_by_stats(initial_masks, num_groups=args.num_groups)
        print(f"  Created {len(grouped_masks)} mask groups")
        
        # Alpha 맵 추출
        print("  Extracting alpha maps from AlphaPredictor...")
        alpha_s, alpha_i = extract_alpha_maps(cidnet_sam_model, input_image, device)
        
        # 시각화 및 alpha 맵 저장
        output_path = os.path.join(analysis_dir, f"{image_count:03d}.png")
        alpha_s_stats, alpha_i_stats = visualize_alpha_with_sam_masks(
            input_image, alpha_s, alpha_i, grouped_masks, output_path,
            output_dir=method1_dir, filename_prefix=f"{image_count:03d}"
        )
    
    print(f"\n✓ Analysis complete! Results saved to: {method1_dir}")
    print(f"  - Alpha_s maps: {alpha_s_dir}")
    print(f"  - Alpha_i maps: {alpha_i_dir}")
    print(f"  - Input images: {input_dir}")
    print(f"  - GT images: {gt_dir}")
    print(f"  - Analysis visualizations: {analysis_dir}")
