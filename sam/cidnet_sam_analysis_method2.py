import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

from sam_metrics import group_masks_by_stats, sort_files_by_number
from utils import extract_alpha_maps, load_sam_model, load_cidnet_sam_model

warnings.filterwarnings("ignore")


def analyze_alpha_variance_by_group(alpha_map, grouped_masks):
    """
    2번 방법: 그룹 내 Alpha 값의 분산 분석
    - 각 SAM 그룹 내에서 alpha 값들의 평균, 표준편차, 변동계수(CV) 계산
    - 그룹 내 분산 vs 그룹 간 분산 비교
    """
    group_stats = []
    
    # Background 통계
    combined_mask = np.zeros_like(grouped_masks[0]['segmentation'], dtype=bool)
    for mask in grouped_masks:
        combined_mask |= mask['segmentation']
    background_mask = ~combined_mask
    
    if np.any(background_mask):
        bg_values = alpha_map[background_mask]
        group_stats.append({
            'group_id': 'Background',
            'mean': np.mean(bg_values),
            'std': np.std(bg_values),
            'cv': np.std(bg_values) / (np.mean(bg_values) + 1e-8),  # Coefficient of Variation
            'min': np.min(bg_values),
            'max': np.max(bg_values),
            'range': np.max(bg_values) - np.min(bg_values),
            'area': np.sum(background_mask),
            'values': bg_values
        })
    
    # 각 그룹별 통계
    for i, mask_group in enumerate(grouped_masks):
        mask = mask_group['segmentation']
        values = alpha_map[mask]
        
        group_stats.append({
            'group_id': i + 1,
            'mean': np.mean(values),
            'std': np.std(values),
            'cv': np.std(values) / (np.mean(values) + 1e-8),
            'min': np.min(values),
            'max': np.max(values),
            'range': np.max(values) - np.min(values),
            'area': mask_group['area'],
            'values': values
        })
    
    # Within-group variance (그룹 내 분산)
    within_group_variance = np.mean([stat['std']**2 for stat in group_stats])
    
    # Between-group variance (그룹 간 분산)
    group_means = [stat['mean'] for stat in group_stats]
    overall_mean = np.mean(alpha_map)
    group_areas = [stat['area'] for stat in group_stats]
    total_area = sum(group_areas)
    between_group_variance = sum([
        (area / total_area) * (mean - overall_mean)**2 
        for mean, area in zip(group_means, group_areas)
    ])
    
    # F-statistic (ANOVA-like)
    f_statistic = between_group_variance / (within_group_variance + 1e-8)
    
    return group_stats, within_group_variance, between_group_variance, f_statistic


def visualize_variance_analysis(image, alpha_s, alpha_i, grouped_masks, 
                                alpha_s_stats, alpha_i_stats,
                                alpha_s_variance, alpha_i_variance,
                                output_path=None):
    """분산 분석 결과 시각화"""
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    alpha_i_mean = np.mean(alpha_i, axis=0)
    
    # Figure 생성
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Method 2: Within-Group Variance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_array)
    ax1.set_title('Original Image', fontsize=12)
    ax1.axis('off')
    
    # 2. SAM Masks
    ax2 = fig.add_subplot(gs[0, 1])
    mask_overlay = np.zeros_like(img_array, dtype=float)
    colors_list = plt.cm.Set1(np.linspace(0, 1, len(grouped_masks)))
    for i, mask_group in enumerate(grouped_masks):
        mask = mask_group['segmentation']
        color = colors_list[i][:3]
        for c in range(3):
            mask_overlay[:, :, c][mask] = color[c]
    ax2.imshow(img_array)
    ax2.imshow(mask_overlay, alpha=0.5)
    ax2.set_title(f'SAM Groups ({len(grouped_masks)} groups)', fontsize=12)
    ax2.axis('off')
    
    # 3. Alpha_s CV (Coefficient of Variation) Map
    ax3 = fig.add_subplot(gs[0, 2])
    cv_map = np.zeros_like(alpha_s)
    for stat in alpha_s_stats:
        if stat['group_id'] == 'Background':
            combined_mask = np.zeros_like(grouped_masks[0]['segmentation'], dtype=bool)
            for mask in grouped_masks:
                combined_mask |= mask['segmentation']
            cv_map[~combined_mask] = stat['cv']
        else:
            mask = grouped_masks[stat['group_id'] - 1]['segmentation']
            cv_map[mask] = stat['cv']
    
    im3 = ax3.imshow(cv_map, cmap='coolwarm', vmin=0, vmax=np.percentile(cv_map, 95))
    ax3.set_title('Alpha_s CV per Group\n(lower=more uniform)', fontsize=12)
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # 4. Alpha_s Box Plot
    ax4 = fig.add_subplot(gs[1, :])
    box_data_s = [stat['values'] for stat in alpha_s_stats]
    box_labels_s = [f"G{stat['group_id']}" for stat in alpha_s_stats]
    bp1 = ax4.boxplot(box_data_s, labels=box_labels_s, patch_artist=True)
    for patch, stat in zip(bp1['boxes'], alpha_s_stats):
        patch.set_facecolor('lightblue')
    ax4.set_xlabel('SAM Group', fontsize=11)
    ax4.set_ylabel('Alpha_s Value', fontsize=11)
    ax4.set_title(f'Alpha_s Distribution by Group\nWithin-group var: {alpha_s_variance[0]:.6f}, Between-group var: {alpha_s_variance[1]:.6f}, F-stat: {alpha_s_variance[2]:.2f}', 
                  fontsize=12)
    ax4.grid(axis='y', alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 5. Alpha_i Box Plot
    ax5 = fig.add_subplot(gs[2, :])
    box_data_i = [stat['values'] for stat in alpha_i_stats]
    box_labels_i = [f"G{stat['group_id']}" for stat in alpha_i_stats]
    bp2 = ax5.boxplot(box_data_i, labels=box_labels_i, patch_artist=True)
    for patch, stat in zip(bp2['boxes'], alpha_i_stats):
        patch.set_facecolor('lightcoral')
    ax5.set_xlabel('SAM Group', fontsize=11)
    ax5.set_ylabel('Alpha_i Value (mean)', fontsize=11)
    ax5.set_title(f'Alpha_i Distribution by Group\nWithin-group var: {alpha_i_variance[0]:.6f}, Between-group var: {alpha_i_variance[1]:.6f}, F-stat: {alpha_i_variance[2]:.2f}', 
                  fontsize=12)
    ax5.grid(axis='y', alpha=0.3)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_variance_analysis_results(alpha_s_stats, alpha_i_stats, 
                                    alpha_s_variance, alpha_i_variance):
    """분산 분석 결과를 상세하게 출력"""
    print("\n" + "="*100)
    print("METHOD 2: WITHIN-GROUP VARIANCE ANALYSIS")
    print("="*100)
    
    print(f"\n{'Group':<12} {'Area':<10} {'Alpha_s Mean':<15} {'Alpha_s Std':<15} {'Alpha_s CV':<15} {'Alpha_s Range':<15}")
    print("-"*100)
    
    for stat in alpha_s_stats:
        group_id = stat['group_id']
        area = stat['area']
        print(f"{str(group_id):<12} {area:<10} {stat['mean']:<15.4f} {stat['std']:<15.4f} {stat['cv']:<15.4f} {stat['range']:<15.4f}")
    
    print(f"\n{'Group':<12} {'Area':<10} {'Alpha_i Mean':<15} {'Alpha_i Std':<15} {'Alpha_i CV':<15} {'Alpha_i Range':<15}")
    print("-"*100)
    
    for stat in alpha_i_stats:
        group_id = stat['group_id']
        area = stat['area']
        print(f"{str(group_id):<12} {area:<10} {stat['mean']:<15.4f} {stat['std']:<15.4f} {stat['cv']:<15.4f} {stat['range']:<15.4f}")
    
    print("\n" + "-"*100)
    print("VARIANCE DECOMPOSITION:")
    print("-"*100)
    
    print(f"\nAlpha_s:")
    print(f"  Within-group variance:  {alpha_s_variance[0]:.6f}")
    print(f"  Between-group variance: {alpha_s_variance[1]:.6f}")
    print(f"  F-statistic:            {alpha_s_variance[2]:.2f}")
    print(f"  Variance ratio (B/W):   {alpha_s_variance[1] / (alpha_s_variance[0] + 1e-8):.4f}")
    
    print(f"\nAlpha_i:")
    print(f"  Within-group variance:  {alpha_i_variance[0]:.6f}")
    print(f"  Between-group variance: {alpha_i_variance[1]:.6f}")
    print(f"  F-statistic:            {alpha_i_variance[2]:.2f}")
    print(f"  Variance ratio (B/W):   {alpha_i_variance[1] / (alpha_i_variance[0] + 1e-8):.4f}")
    
    print("\n" + "-"*100)
    print("INTERPRETATION:")
    print("-"*100)
    print("• Higher F-statistic = Alpha values are more similar within SAM groups")
    print("• Higher variance ratio = Between-group differences dominate over within-group variation")
    print("• Lower CV (Coefficient of Variation) = More uniform values within a group")
    print("="*100 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Method 2: Variance analysis of AlphaPredictor vs SAM grouping')
    parser.add_argument('--model_path', type=str, default="weights/train2025-10-13-005336/epoch_760.pth",
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
    method2_dir = os.path.join(args.output_dir, "method2_variance_analysis")
    os.makedirs(method2_dir, exist_ok=True)
    
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
    
    # 결과 저장용 리스트
    all_results = []
    
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
        alpha_i_mean = np.mean(alpha_i, axis=0)
        
        # 분산 분석
        print("  Analyzing variance...")
        alpha_s_stats, s_within, s_between, s_f = analyze_alpha_variance_by_group(alpha_s, grouped_masks)
        alpha_i_stats, i_within, i_between, i_f = analyze_alpha_variance_by_group(alpha_i_mean, grouped_masks)
        
        alpha_s_variance = (s_within, s_between, s_f)
        alpha_i_variance = (i_within, i_between, i_f)
        
        # 시각화
        output_path = os.path.join(method2_dir, f"{input_filename}_variance_analysis.png")
        visualize_variance_analysis(
            input_image, alpha_s, alpha_i, grouped_masks,
            alpha_s_stats, alpha_i_stats,
            alpha_s_variance, alpha_i_variance,
            output_path
        )
        
        # 결과 출력
        print_variance_analysis_results(alpha_s_stats, alpha_i_stats, 
                                       alpha_s_variance, alpha_i_variance)
        
        # 결과 저장
        all_results.append({
            'filename': input_filename,
            'alpha_s_f_stat': s_f,
            'alpha_i_f_stat': i_f,
            'alpha_s_variance_ratio': s_between / (s_within + 1e-8),
            'alpha_i_variance_ratio': i_between / (i_within + 1e-8),
            'alpha_s_within_var': s_within,
            'alpha_s_between_var': s_between,
            'alpha_i_within_var': i_within,
            'alpha_i_between_var': i_between
        })
    
    # 전체 결과 요약
    print("\n" + "="*100)
    print("SUMMARY ACROSS ALL IMAGES")
    print("="*100)
    
    df = pd.DataFrame(all_results)
    
    print(f"\nAlpha_s F-statistic:     Mean={df['alpha_s_f_stat'].mean():.2f}, Std={df['alpha_s_f_stat'].std():.2f}")
    print(f"Alpha_i F-statistic:     Mean={df['alpha_i_f_stat'].mean():.2f}, Std={df['alpha_i_f_stat'].std():.2f}")
    print(f"\nAlpha_s Variance Ratio:  Mean={df['alpha_s_variance_ratio'].mean():.4f}, Std={df['alpha_s_variance_ratio'].std():.4f}")
    print(f"Alpha_i Variance Ratio:  Mean={df['alpha_i_variance_ratio'].mean():.4f}, Std={df['alpha_i_variance_ratio'].std():.4f}")
    
    # CSV로 저장
    csv_path = os.path.join(method2_dir, "variance_analysis_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Summary saved to: {csv_path}")
    
    print(f"\n✓ Analysis complete! Results saved to: {method2_dir}")
