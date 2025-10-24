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
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pandas as pd

from sam_metrics import group_masks_by_stats, sort_files_by_number
from utils import extract_alpha_maps, load_sam_model, load_cidnet_sam_model

warnings.filterwarnings("ignore")


def create_sam_label_map(grouped_masks, image_shape):
    """SAM 그룹을 label map으로 변환"""
    label_map = np.zeros(image_shape[:2], dtype=int)
    
    # Background는 0
    for i, mask_group in enumerate(grouped_masks):
        mask = mask_group['segmentation']
        label_map[mask] = i + 1
    
    return label_map


def cluster_alpha_values(alpha_s, alpha_i, num_clusters, spatial_weight=0.0):
    """
    Alpha 값을 기반으로 K-means 클러스터링
    
    Args:
        alpha_s: (H, W) alpha_s map
        alpha_i: (3, H, W) alpha_i map
        num_clusters: 클러스터 개수
        spatial_weight: 공간 정보 가중치 (0~1, 0이면 순수 alpha 값만 사용)
    """
    h, w = alpha_s.shape
    alpha_i_mean = np.mean(alpha_i, axis=0)
    
    # Feature 구성
    features = []
    
    # Alpha 값
    features.append(alpha_s.reshape(-1, 1))
    features.append(alpha_i_mean.reshape(-1, 1))
    
    # 선택적으로 공간 정보 추가 (normalized coordinates)
    if spatial_weight > 0:
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        y_coords = y_coords.reshape(-1, 1) / h * spatial_weight
        x_coords = x_coords.reshape(-1, 1) / w * spatial_weight
        features.append(y_coords)
        features.append(x_coords)
    
    # Concatenate features
    X = np.hstack(features)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Reshape to image
    label_map = labels.reshape(h, w)
    
    return label_map, kmeans


def calculate_iou(mask1, mask2):
    """Calculate IoU between two binary masks"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def calculate_dice(mask1, mask2):
    """Calculate Dice coefficient between two binary masks"""
    intersection = np.logical_and(mask1, mask2).sum()
    sum_masks = mask1.sum() + mask2.sum()
    if sum_masks == 0:
        return 0.0
    return 2 * intersection / sum_masks


def compare_segmentations(sam_labels, alpha_labels):
    """
    SAM 그룹핑과 Alpha 클러스터링의 유사도 측정
    
    Returns:
        - ARI (Adjusted Rand Index): -1 ~ 1, 높을수록 유사
        - NMI (Normalized Mutual Information): 0 ~ 1, 높을수록 유사
        - Mean IoU: 평균 IoU (헝가리안 매칭 사용)
        - Mean Dice: 평균 Dice coefficient
    """
    # Flatten labels
    sam_flat = sam_labels.flatten()
    alpha_flat = alpha_labels.flatten()
    
    # ARI and NMI
    ari = adjusted_rand_score(sam_flat, alpha_flat)
    nmi = normalized_mutual_info_score(sam_flat, alpha_flat)
    
    # Calculate IoU and Dice using Hungarian matching
    num_sam_groups = len(np.unique(sam_labels))
    num_alpha_groups = len(np.unique(alpha_labels))
    
    # Create IoU and Dice matrices
    iou_matrix = np.zeros((num_sam_groups, num_alpha_groups))
    dice_matrix = np.zeros((num_sam_groups, num_alpha_groups))
    
    for i, sam_id in enumerate(np.unique(sam_labels)):
        sam_mask = (sam_labels == sam_id)
        for j, alpha_id in enumerate(np.unique(alpha_labels)):
            alpha_mask = (alpha_labels == alpha_id)
            iou_matrix[i, j] = calculate_iou(sam_mask, alpha_mask)
            dice_matrix[i, j] = calculate_dice(sam_mask, alpha_mask)
    
    # Hungarian matching (greedy: pick best match for each SAM group)
    matched_ious = []
    matched_dices = []
    used_alpha_ids = set()
    
    for i in range(num_sam_groups):
        best_j = -1
        best_iou = 0
        for j in range(num_alpha_groups):
            if j not in used_alpha_ids and iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_j = j
        
        if best_j >= 0:
            matched_ious.append(iou_matrix[i, best_j])
            matched_dices.append(dice_matrix[i, best_j])
            used_alpha_ids.add(best_j)
    
    mean_iou = np.mean(matched_ious) if matched_ious else 0.0
    mean_dice = np.mean(matched_dices) if matched_dices else 0.0
    
    return {
        'ari': ari,
        'nmi': nmi,
        'mean_iou': mean_iou,
        'mean_dice': mean_dice,
        'iou_matrix': iou_matrix,
        'dice_matrix': dice_matrix
    }


def visualize_clustering_comparison(image, sam_labels, alpha_labels_pure, alpha_labels_spatial,
                                    metrics_pure, metrics_spatial, output_path=None):
    """클러스터링 비교 시각화"""
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    # Figure 생성
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle('Method 3: Clustering Comparison (SAM vs Alpha-based)', fontsize=16, fontweight='bold')
    
    # Color mapping
    num_sam_groups = len(np.unique(sam_labels))
    num_alpha_groups = len(np.unique(alpha_labels_pure))
    
    cmap = plt.cm.get_cmap('tab20', max(num_sam_groups, num_alpha_groups))
    
    # 1. Original Image
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')
    
    # 2. SAM Grouping
    axes[0, 1].imshow(sam_labels, cmap=cmap, interpolation='nearest')
    axes[0, 1].set_title(f'SAM Grouping ({num_sam_groups} groups)', fontsize=12)
    axes[0, 1].axis('off')
    
    # 3. Alpha Clustering (Pure)
    axes[0, 2].imshow(alpha_labels_pure, cmap=cmap, interpolation='nearest')
    axes[0, 2].set_title(f'Alpha Clustering - Pure\n({num_alpha_groups} clusters)', fontsize=12)
    axes[0, 2].axis('off')
    
    # 4. SAM + Alpha Pure Overlay
    axes[1, 0].imshow(img_array)
    sam_edges = ndimage.binary_dilation(ndimage.sobel(sam_labels) > 0)
    alpha_edges = ndimage.binary_dilation(ndimage.sobel(alpha_labels_pure) > 0)
    
    edge_overlay = np.zeros((*sam_labels.shape, 3))
    edge_overlay[sam_edges] = [1, 0, 0]  # Red for SAM
    edge_overlay[alpha_edges] = [0, 1, 0]  # Green for Alpha
    edge_overlay[sam_edges & alpha_edges] = [1, 1, 0]  # Yellow for overlap
    
    axes[1, 0].imshow(edge_overlay, alpha=0.7)
    axes[1, 0].set_title(f'Overlay: SAM (Red) vs Alpha-Pure (Green)\nARI={metrics_pure["ari"]:.3f}, NMI={metrics_pure["nmi"]:.3f}', fontsize=11)
    axes[1, 0].axis('off')
    
    # 5. Alpha Clustering (Spatial)
    axes[1, 1].imshow(alpha_labels_spatial, cmap=cmap, interpolation='nearest')
    axes[1, 1].set_title(f'Alpha Clustering - Spatial\n({len(np.unique(alpha_labels_spatial))} clusters)', fontsize=12)
    axes[1, 1].axis('off')
    
    # 6. Metrics Comparison
    axes[1, 2].axis('off')
    
    metrics_text = "Similarity Metrics:\n\n"
    metrics_text += "Pure Alpha Clustering:\n"
    metrics_text += f"  ARI:       {metrics_pure['ari']:.4f}\n"
    metrics_text += f"  NMI:       {metrics_pure['nmi']:.4f}\n"
    metrics_text += f"  Mean IoU:  {metrics_pure['mean_iou']:.4f}\n"
    metrics_text += f"  Mean Dice: {metrics_pure['mean_dice']:.4f}\n\n"
    
    metrics_text += "Spatial Alpha Clustering:\n"
    metrics_text += f"  ARI:       {metrics_spatial['ari']:.4f}\n"
    metrics_text += f"  NMI:       {metrics_spatial['nmi']:.4f}\n"
    metrics_text += f"  Mean IoU:  {metrics_spatial['mean_iou']:.4f}\n"
    metrics_text += f"  Mean Dice: {metrics_spatial['mean_dice']:.4f}\n\n"
    
    metrics_text += "Interpretation:\n"
    metrics_text += "• ARI, NMI closer to 1 = more similar\n"
    metrics_text += "• IoU, Dice closer to 1 = better overlap\n"
    metrics_text += "• Pure uses only alpha values\n"
    metrics_text += "• Spatial includes pixel location"
    
    axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                   family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_clustering_results(metrics_pure, metrics_spatial):
    """클러스터링 비교 결과 출력"""
    print("\n" + "="*100)
    print("METHOD 3: CLUSTERING COMPARISON")
    print("="*100)
    
    print("\n" + "-"*100)
    print("SIMILARITY METRICS (SAM Grouping vs Alpha-based Clustering)")
    print("-"*100)
    
    print(f"\n{'Metric':<25} {'Pure Alpha':<20} {'Spatial Alpha':<20} {'Difference':<20}")
    print("-"*100)
    
    metrics = ['ari', 'nmi', 'mean_iou', 'mean_dice']
    metric_names = ['ARI (Rand Index)', 'NMI (Mutual Info)', 'Mean IoU', 'Mean Dice Coefficient']
    
    for metric, name in zip(metrics, metric_names):
        pure_val = metrics_pure[metric]
        spatial_val = metrics_spatial[metric]
        diff = spatial_val - pure_val
        
        print(f"{name:<25} {pure_val:<20.4f} {spatial_val:<20.4f} {diff:+.4f}")
    
    print("\n" + "-"*100)
    print("INTERPRETATION:")
    print("-"*100)
    print("• ARI (Adjusted Rand Index): Measures clustering similarity (-1 to 1, higher is better)")
    print("• NMI (Normalized Mutual Information): Measures information shared (0 to 1, higher is better)")
    print("• Mean IoU: Average Intersection over Union of matched groups (0 to 1, higher is better)")
    print("• Mean Dice: Average Dice coefficient of matched groups (0 to 1, higher is better)")
    print("\n• Pure Alpha: Clustering based only on alpha_s and alpha_i values")
    print("• Spatial Alpha: Clustering with alpha values + spatial coordinates")
    print("\nHigher values indicate that Alpha predictions are more consistent with SAM grouping")
    print("="*100 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Method 3: Clustering comparison of AlphaPredictor vs SAM grouping')
    parser.add_argument('--model_path', type=str, default="weights/train2025-10-13-005336/epoch_760.pth",
                        help='Path to CIDNet_sam model checkpoint')
    parser.add_argument('--dir', type=str, default="datasets/LOLdataset/eval15",
                        help='Base directory containing low/high subdirectories')
    parser.add_argument('--output_dir', type=str, default="sam/analysis_results",
                        help='Directory to save analysis results')
    parser.add_argument('--sam_model', type=str, default="Gourieff/ReActor/models/sams/sam_vit_b_01ec64.pth",
                        help='SAM model path')
    parser.add_argument('--num_groups', type=int, default=10,
                        help='Number of groups for SAM and clusters for Alpha')
    parser.add_argument('--spatial_weight', type=float, default=0.3,
                        help='Weight for spatial information in clustering (0=pure alpha, 1=equal weight)')
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
    method3_dir = os.path.join(args.output_dir, "method3_clustering_comparison")
    os.makedirs(method3_dir, exist_ok=True)
    
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
        sam_labels = create_sam_label_map(grouped_masks, np.array(input_image).shape)
        print(f"  Created {len(grouped_masks)} SAM groups")
        
        # Alpha 맵 추출
        print("  Extracting alpha maps from AlphaPredictor...")
        alpha_s, alpha_i = extract_alpha_maps(cidnet_sam_model, input_image, device)
        
        # Alpha 기반 클러스터링 (Pure)
        print("  Clustering based on alpha values (pure)...")
        alpha_labels_pure, kmeans_pure = cluster_alpha_values(
            alpha_s, alpha_i, num_clusters=args.num_groups, spatial_weight=0.0
        )
        
        # Alpha 기반 클러스터링 (Spatial)
        print("  Clustering based on alpha values (spatial)...")
        alpha_labels_spatial, kmeans_spatial = cluster_alpha_values(
            alpha_s, alpha_i, num_clusters=args.num_groups, spatial_weight=args.spatial_weight
        )
        
        # 유사도 측정
        print("  Calculating similarity metrics...")
        metrics_pure = compare_segmentations(sam_labels, alpha_labels_pure)
        metrics_spatial = compare_segmentations(sam_labels, alpha_labels_spatial)
        
        # 시각화
        output_path = os.path.join(method3_dir, f"{input_filename}_clustering_comparison.png")
        visualize_clustering_comparison(
            input_image, sam_labels, alpha_labels_pure, alpha_labels_spatial,
            metrics_pure, metrics_spatial, output_path
        )
        
        # 결과 출력
        print_clustering_results(metrics_pure, metrics_spatial)
        
        # 결과 저장
        all_results.append({
            'filename': input_filename,
            'pure_ari': metrics_pure['ari'],
            'pure_nmi': metrics_pure['nmi'],
            'pure_mean_iou': metrics_pure['mean_iou'],
            'pure_mean_dice': metrics_pure['mean_dice'],
            'spatial_ari': metrics_spatial['ari'],
            'spatial_nmi': metrics_spatial['nmi'],
            'spatial_mean_iou': metrics_spatial['mean_iou'],
            'spatial_mean_dice': metrics_spatial['mean_dice']
        })
    
    # 전체 결과 요약
    print("\n" + "="*100)
    print("SUMMARY ACROSS ALL IMAGES")
    print("="*100)
    
    df = pd.DataFrame(all_results)
    
    print("\nPure Alpha Clustering:")
    print(f"  ARI:       Mean={df['pure_ari'].mean():.4f}, Std={df['pure_ari'].std():.4f}")
    print(f"  NMI:       Mean={df['pure_nmi'].mean():.4f}, Std={df['pure_nmi'].std():.4f}")
    print(f"  Mean IoU:  Mean={df['pure_mean_iou'].mean():.4f}, Std={df['pure_mean_iou'].std():.4f}")
    print(f"  Mean Dice: Mean={df['pure_mean_dice'].mean():.4f}, Std={df['pure_mean_dice'].std():.4f}")
    
    print("\nSpatial Alpha Clustering:")
    print(f"  ARI:       Mean={df['spatial_ari'].mean():.4f}, Std={df['spatial_ari'].std():.4f}")
    print(f"  NMI:       Mean={df['spatial_nmi'].mean():.4f}, Std={df['spatial_nmi'].std():.4f}")
    print(f"  Mean IoU:  Mean={df['spatial_mean_iou'].mean():.4f}, Std={df['spatial_mean_iou'].std():.4f}")
    print(f"  Mean Dice: Mean={df['spatial_mean_dice'].mean():.4f}, Std={df['spatial_mean_dice'].std():.4f}")
    
    # CSV로 저장
    csv_path = os.path.join(method3_dir, "clustering_comparison_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Summary saved to: {csv_path}")
    
    print(f"\n✓ Analysis complete! Results saved to: {method3_dir}")
