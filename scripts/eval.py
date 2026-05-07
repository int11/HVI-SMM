import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import *
from loss import *
from scripts.measure import metrics, metrics_no_ref
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
from torch.utils.data import DataLoader
from data.eval_sets import SingleFolderEvalDataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def eval(model, data, alpha_combinations, unpaired=False):
    """
    Unified evaluation function for both CIDNet and CIDNet_SMM models.
    Efficiently handles multiple alpha combinations by computing features once,
    then applying different alpha values without redundant forward passes.

    Args:
        model: BaseCIDNet or BaseCIDNet_SMM model instance
        data: DataLoader for test data
        alpha_combinations: List of (base_alpha_s, base_alpha_i, alpha_rgb) tuples
        unpaired: bool, if True, skip GT processing (default: False)
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

        if not unpaired:
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

            # Convert to numpy and store1
            output_rgb = torch.clamp(output_rgb, 0, 1)
            output_np = output_rgb.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            if not unpaired:
                gt_pil = ToPILImage()(gt_img.squeeze(0).cpu())
                results[(base_alpha_s, base_alpha_i, alpha_rgb)][1].append(gt_pil)
            else:
                results[(base_alpha_s, base_alpha_i, alpha_rgb)][1].append(None)

            results[(base_alpha_s, base_alpha_i, alpha_rgb)][0].append(output_np)

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


def make_row(dataset, model, alpha='-', psnr='-', ssim='-', lpips='-', niqe='-', brisque='-'):
    return {'Dataset': dataset, 'Model': model, 'Alpha': alpha,
            'PSNR': psnr, 'SSIM': ssim, 'LPIPS': lpips, 'NIQE': niqe, 'BRISQUE': brisque}


def print_table(data):
    header = (f"{'Dataset':<20} | {'Model':<12} | {'Alpha':<16} | "
              f"{'PSNR':>8} | {'SSIM':>8} | {'LPIPS':>8} | {'NIQE':>8} | {'BRISQUE':>8}")
    sep = "-" * len(header)
    print("\n\n" + "!" * 60)
    print("                 FINAL EVALUATION SUMMARY")
    print("!" * 60)
    print(header)
    print(sep)
    for item in data:
        print(f"{item['Dataset']:<20} | {item['Model']:<12} | {item['Alpha']:<16} | "
              f"{item['PSNR']:>8} | {item['SSIM']:>8} | {item['LPIPS']:>8} | "
              f"{item['NIQE']:>8} | {item['BRISQUE']:>8}")
        if item['Model'] == 'CIDNet SMM':
            print(sep)
    print("!" * 60)


def make_grid_comparison_image(rows_data, col_labels, header_h=40, row_label_w=180):
    """
    rows_data: list of (row_label_str, [PIL.Image, ...]) — one entry per alpha combo
    col_labels: list of column header strings
    Returns a grid: rows=alpha combos, cols=views
    """
    n_rows = len(rows_data)
    n_cols = len(rows_data[0][1])
    w, h = rows_data[0][1][0].size
    combined = Image.new('RGB', (row_label_w + w * n_cols, header_h + h * n_rows), (255, 255, 255))
    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    for c, label in enumerate(col_labels):
        draw.text((row_label_w + w * c + 5, 8), label, fill=(0, 0, 0), font=font)
    for r, (row_label, images) in enumerate(rows_data):
        y = header_h + h * r
        draw.text((5, y + h // 2 - 10), row_label, fill=(0, 0, 0), font=font)
        for c, img in enumerate(images):
            combined.paste(img, (row_label_w + w * c, y))
    return combined


if __name__ == '__main__':
    parser = option()
    parser.add_argument('--input_image', type=str, default=None, help='Path to input image')
    parser.add_argument('--cidnet_smm_path', type=str, default="weights/lol_v1/v5/20260104_205240_gt_mean_loss='hvi'/epoch_600.pth", help='Path to the CIDNet SMM model weights')
    parser.add_argument('--output_dir', type=str, default='results/smm_lolv1', help='Directory to save comparison images')
    # Available CIDNet models from Hugging Face:
    #   Fediory/HVI-CIDNet
    #   Fediory/HVI-CIDNet-LOLv1-wperc
    #   Fediory/HVI-CIDNet-LOLv1-woperc
    #   Fediory/HVI-CIDNet-LOLv2-real-bestPSNR
    #   Fediory/HVI-CIDNet-LOLv2-real-bestSSIM
    #   Fediory/HVI-CIDNet-LOLv2-syn-wperc
    #   Fediory/HVI-CIDNet-LOLv2-syn-woperc
    #   Fediory/HVI-CIDNet-Generalization
    #   Fediory/HVI-CIDNet-LOL-Blur
    #   Fediory/HVI-CIDNet-SICE
    #   Fediory/HVI-CIDNet-Sony-Total-Dark
    #   Fediory/HVI-CIDNet-FiveK
    parser.add_argument('--cidnet_path', type=str, default="Fediory/HVI-CIDNet-LOLv1-wperc", help='Path to the base CIDNet model')
    parser.add_argument('--alpha_combinations', type=str, nargs='+',
                        default=['1.0,1.0,1.0', '0.7,1.0,1.0', '1.3,1.0,1.0','1.6,1.0,1.0','1.9,1.0,1.0','2.1,1.0,1.0', '1.0,0.7,1.0', '1.0,1.3,1.0', '1.0,1.6,1.0', '1.0,1.9,1.0', '1.0,2.1,1.0'],
                        help='Alpha combinations as "alpha_s,alpha_i,alpha_rgb" (space-separated). '
                             'Example: --alpha_combinations 1.3,1.0,1.0 1.0,1.0,0.8 1.0,1.0,1.0')
    parser.add_argument('--use_gt_mean', type=bool, default=False, help='Use the mean of GT to rectify the output of the model')
    parser.add_argument('--unpaired_dataset_dir', type=str, default='./datasets/unpaired', help='Directory path containing unpaired datasets')
    parser.add_argument('--unpaired_dataset_names', type=str, nargs='+', default=['DICM', 'LIME', 'MEF', 'NPE'], help='List of unpaired dataset folder names to evaluate')
    args = parser.parse_args()

    # Parse alpha_combinations from "a_s,a_i,a_rgb" strings
    alpha_combinations = []
    for combo_str in args.alpha_combinations:
        parts = [float(x) for x in combo_str.split(',')]
        assert len(parts) == 3, f"Each alpha combination must have 3 values: {combo_str}"
        alpha_combinations.append(tuple(parts))

    # ── Data loader setup ──────────────────────────────────────────────────
    if args.input_image is not None:
        input_img = Image.open(args.input_image)
        input_tensor = ToTensor()(input_img).unsqueeze(0)
        data_loader = [(input_tensor, torch.zeros_like(input_tensor))]
    else:
        training_data_loader, testing_data_loader = load_datasets(args)
        data_loader = testing_data_loader

    # ── Load CIDNet_SMM & register hook ───────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_net = CIDNet_SMM().cuda()
    eval_net.load_state_dict(
        torch.load(args.cidnet_smm_path, map_location='cpu')['model_state_dict']
    )
    print(f"Loaded CIDNet_SMM checkpoint from {args.cidnet_smm_path}")

    alpha_outputs = []
    hook = eval_net.alpha_predictor.register_forward_hook(
        lambda m, i, o: alpha_outputs.append(o)
    )

    results = eval(eval_net, data_loader, alpha_combinations)
    hook.remove()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Single-image mode ─────────────────────────────────────────────────
    if args.input_image is not None:
        for combo in alpha_combinations:
            output_list, _ = results[combo]
            combo_str = f"alpha_{combo[0]}_{combo[1]}_{combo[2]}"
            out_path = os.path.join(args.output_dir, f'{combo_str}_output.png')
            Image.fromarray((output_list[0] * 255).astype(np.uint8)).save(out_path)
            print(f"Saved [{combo_str}] output to {out_path}")

        if alpha_outputs:
            sf_out = alpha_outputs[0]  # shape: (1, 2, h, w)
            for ch, name in [(0, 'alpha_s'), (1, 'alpha_i')]:
                array_to_heatmap(sf_out[0, ch].cpu().numpy()).save(
                    os.path.join(args.output_dir, f'single_image_{name}.png')
                )
            print(f"Saved alpha maps to {args.output_dir}")

    # ── Dataset mode ──────────────────────────────────────────────────────
    else:
        # Load base CIDNet
        cidnet_base = load_cidnet_base_model(args.cidnet_path, device)
        print(f"Loaded base CIDNet model from {args.cidnet_path}")

        # Extract per-image alpha maps (batch_size, 2, h, w) → flat lists
        alpha_s_list = [out[b, 0].cpu().numpy() for out in alpha_outputs for b in range(out.shape[0])]
        alpha_i_list = [out[b, 1].cpu().numpy() for out in alpha_outputs for b in range(out.shape[0])]

        # Evaluate base CIDNet for all combinations
        results_base = eval(cidnet_base, data_loader, alpha_combinations)
        gt_list = results[alpha_combinations[0]][1]  # GT is same across all combinations

        # Paired metrics per combination
        dataset_name = args.dataset.upper() if args.dataset else 'Paired'
        final_summary_data = []

        for combo in alpha_combinations:
            output_smm, _ = results[combo]
            output_base, _ = results_base[combo]
            psnr_smm, ssim_smm, lpips_smm = metrics(output_smm, gt_list, use_gt_mean=args.use_gt_mean)
            psnr_base, ssim_base, lpips_base = metrics(output_base, gt_list, use_gt_mean=args.use_gt_mean)
            alpha_str = f"({combo[0]},{combo[1]},{combo[2]})"
            final_summary_data += [
                make_row(f'Paired ({dataset_name})', 'Base CIDNet', alpha=alpha_str,
                         psnr=f"{psnr_base:.4f}", ssim=f"{ssim_base:.4f}", lpips=f"{lpips_base:.4f}"),
                make_row(f'Paired ({dataset_name})', 'CIDNet SMM', alpha=alpha_str,
                         psnr=f"{psnr_smm:.4f}", ssim=f"{ssim_smm:.4f}", lpips=f"{lpips_smm:.4f}"),
            ]

        # ── Unpaired datasets ──────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("Unpaired Dataset Evaluation")
        print("=" * 60)

        total_imgs = 0
        total_niqe = {combo: {'smm': 0.0, 'base': 0.0} for combo in alpha_combinations}
        total_brisque = {combo: {'smm': 0.0, 'base': 0.0} for combo in alpha_combinations}

        for ds_name in args.unpaired_dataset_names:
            ds_path = os.path.join(args.unpaired_dataset_dir, ds_name)
            if not os.path.exists(ds_path):
                print(f"Skipping {ds_name}: path not found ({ds_path})")
                continue

            print(f"-- Evaluating {ds_name}...", end=" ", flush=True)
            ds_set = SingleFolderEvalDataset(ds_path, transform=ToTensor())
            ds_loader = DataLoader(dataset=ds_set, num_workers=1, batch_size=1, shuffle=False)

            results_smm_ds = eval(eval_net, ds_loader, alpha_combinations, unpaired=True)
            results_base_ds = eval(cidnet_base, ds_loader, alpha_combinations, unpaired=True)
            print("Done")

            n = len(ds_set)
            total_imgs += n

            for combo in alpha_combinations:
                outputs_smm = results_smm_ds[combo][0]
                outputs_base = results_base_ds[combo][0]
                niqe_smm, brisque_smm = metrics_no_ref(outputs_smm)
                niqe_base, brisque_base = metrics_no_ref(outputs_base)

                alpha_str = f"({combo[0]},{combo[1]},{combo[2]})"
                final_summary_data += [
                    make_row(ds_name, 'Base CIDNet', alpha=alpha_str,
                             niqe=f"{niqe_base:.4f}", brisque=f"{brisque_base:.4f}"),
                    make_row(ds_name, 'CIDNet SMM', alpha=alpha_str,
                             niqe=f"{niqe_smm:.4f}", brisque=f"{brisque_smm:.4f}"),
                ]
                total_niqe[combo]['smm']    += niqe_smm * n
                total_niqe[combo]['base']   += niqe_base * n
                total_brisque[combo]['smm'] += brisque_smm * n
                total_brisque[combo]['base'] += brisque_base * n

            # Save unpaired comparison grid (rows=alpha combos, cols=Input/Base/SMM)
            ds_inputs = [
                (ds_set[i][0] if isinstance(ds_set[i], tuple) else ds_set[i]).numpy().transpose(1, 2, 0)
                for i in range(n)
            ]
            cmp_ds_dir = os.path.join(args.output_dir, ds_name, 'comparison')
            os.makedirs(cmp_ds_dir, exist_ok=True)
            for i, in_np in enumerate(ds_inputs):
                input_pil = Image.fromarray((in_np * 255).astype(np.uint8))
                rows_data = []
                for combo in alpha_combinations:
                    row_label = f"α=({combo[0]},{combo[1]},{combo[2]})"
                    out_smm = results_smm_ds[combo][0][i]
                    out_base = results_base_ds[combo][0][i]
                    rows_data.append((row_label, [
                        input_pil,
                        Image.fromarray((out_base * 255).astype(np.uint8)),
                        Image.fromarray((out_smm * 255).astype(np.uint8)),
                    ]))
                make_grid_comparison_image(rows_data, ["Input (No GT)", "Base CIDNet", "CIDNet SMM"]).save(
                    os.path.join(cmp_ds_dir, f'{ds_name}_{i+1:03d}_cmp.png')
                )

        # Unpaired average rows
        if total_imgs > 0:
            for combo in alpha_combinations:
                alpha_str = f"({combo[0]},{combo[1]},{combo[2]})"
                final_summary_data += [
                    make_row('AVERAGE (Unpaired)', 'Base CIDNet', alpha=alpha_str,
                             niqe=f"{total_niqe[combo]['base']/total_imgs:.4f}",
                             brisque=f"{total_brisque[combo]['base']/total_imgs:.4f}"),
                    make_row('AVERAGE (Unpaired)', 'CIDNet SMM', alpha=alpha_str,
                             niqe=f"{total_niqe[combo]['smm']/total_imgs:.4f}",
                             brisque=f"{total_brisque[combo]['smm']/total_imgs:.4f}"),
                ]

        print_table(final_summary_data)

        # ── Save paired images (individual per combo) + grid comparison ──────
        input_images = [batch[0].squeeze(0).numpy().transpose(1, 2, 0) for batch in testing_data_loader]

        indiv_dirs = ['input', 'cidnet', 'cidnet_smm', 'gt', 'alpha_s', 'alpha_i']
        combo_dir_paths = {}
        for combo in alpha_combinations:
            combo_str = f"alpha_{combo[0]}_{combo[1]}_{combo[2]}"
            combo_dir_paths[combo] = {d: os.path.join(args.output_dir, combo_str, d) for d in indiv_dirs}
            for path in combo_dir_paths[combo].values():
                os.makedirs(path, exist_ok=True)

        cmp_dir = os.path.join(args.output_dir, 'comparison')
        os.makedirs(cmp_dir, exist_ok=True)

        col_labels = ["Input", "Base CIDNet", "CIDNet SMM", "GT", "Alpha_s", "Alpha_i"]
        n_imgs = len(results[alpha_combinations[0]][0])
        for idx in range(n_imgs):
            in_np = input_images[idx]
            a_s = alpha_s_list[idx]
            a_i = alpha_i_list[idx]
            input_pil   = Image.fromarray((in_np * 255).astype(np.uint8))
            alpha_s_pil = array_to_heatmap(a_s)
            alpha_i_pil = array_to_heatmap(a_i)
            img_name = f'{idx+1:03d}.png'

            rows_data = []
            for combo in alpha_combinations:
                out_np  = results[combo][0][idx]
                gt_img  = results[combo][1][idx]
                base_np = results_base[combo][0][idx]
                base_pil = Image.fromarray((base_np * 255).astype(np.uint8))
                smm_pil  = Image.fromarray((out_np  * 255).astype(np.uint8))

                dp = combo_dir_paths[combo]
                input_pil.save(os.path.join(dp['input'],      img_name))
                base_pil .save(os.path.join(dp['cidnet'],     img_name))
                smm_pil  .save(os.path.join(dp['cidnet_smm'], img_name))
                gt_img   .save(os.path.join(dp['gt'],         img_name))
                alpha_s_pil.save(os.path.join(dp['alpha_s'],  img_name))
                alpha_i_pil.save(os.path.join(dp['alpha_i'],  img_name))

                row_label = f"α=({combo[0]},{combo[1]},{combo[2]})"
                rows_data.append((row_label, [input_pil, base_pil, smm_pil, gt_img, alpha_s_pil, alpha_i_pil]))

            make_grid_comparison_image(rows_data, col_labels).save(
                os.path.join(cmp_dir, f'comparison_{idx+1:03d}.png')
            )
            print(f"Saved grid comparison [{idx+1:03d}]")

        print(f"\n✓ Total output saved to: {args.output_dir}")
