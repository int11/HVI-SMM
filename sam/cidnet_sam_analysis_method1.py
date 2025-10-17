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


def visualize_alpha_with_sam_masks(image, alpha_s, alpha_i, grouped_masks, output_path=None):
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
    parser = argparse.ArgumentParser(description='Analyze CIDNet_sam AlphaPredictor predictions vs SAM grouping')
    parser.add_argument('--model_path', type=str, default="weights/train2025-10-13-005336/epoch_1500.pth",
                        help='Path to CIDNet_sam model checkpoint')
    parser.add_argument('--dir', type=str, default="datasets/LOLdataset/eval15",
                        help='Base directory containing low/high subdirectories')
    parser.add_argument('--output_dir', type=str, default="sam/analysis_results",
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
    
    # 이미지 디렉토리
    input_dir = os.path.join(args.dir, "low")
    
    # 이미지 파일 리스트
    input_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    input_files = sort_files_by_number(input_files)
    
    if args.max_images:
        input_files = input_files[:args.max_images]
    
    print(f"Processing {len(input_files)} images...\n")
    
    # 모델 로드
    cidnet_sam_model = load_cidnet_sam_model(args.model_path, device)
    sam_model = load_sam_model(args.sam_model, device)
    
    # 각 이미지 처리
    for idx, input_file in enumerate(input_files):
        print(f"\n[{idx+1}/{len(input_files)}] Processing {input_file}...")
        input_filename = os.path.splitext(os.path.basename(input_file))[0]
        
        # 이미지 로드
        input_path = os.path.join(input_dir, input_file)
        input_image = Image.open(input_path).convert('RGB')
        
        # SAM 마스크 생성 및 그룹핑
        print("  Generating SAM masks...")
        initial_masks = sam_model.generate(np.array(input_image))
        grouped_masks = group_masks_by_stats(initial_masks, num_groups=args.num_groups)
        print(f"  Created {len(grouped_masks)} mask groups")
        
        # Alpha 맵 추출
        print("  Extracting alpha maps from AlphaPredictor...")
        alpha_s, alpha_i = extract_alpha_maps(cidnet_sam_model, input_image, device)
        
        # 시각화
        output_path = os.path.join(method1_dir, f"{input_filename}_analysis.png")
        alpha_s_stats, alpha_i_stats = visualize_alpha_with_sam_masks(
            input_image, alpha_s, alpha_i, grouped_masks, output_path
        )
    
    print(f"\n✓ Analysis complete! Results saved to: {method1_dir}")
