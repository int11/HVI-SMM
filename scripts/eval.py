import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data import *
from loss.losses import *
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
from data.eval_sets import SICEDatasetFromFolderEval
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
            
            # Convert to numpy and store
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


def make_row(dataset, model, psnr='-', ssim='-', lpips='-', niqe='-', brisque='-'):
    return {'Dataset': dataset, 'Model': model,
            'PSNR': psnr, 'SSIM': ssim, 'LPIPS': lpips, 'NIQE': niqe, 'BRISQUE': brisque}


def print_table(data):
    header = f"{'Dataset':<20} | {'Model':<12} | {'PSNR':>8} | {'SSIM':>8} | {'LPIPS':>8} | {'NIQE':>8} | {'BRISQUE':>8}"
    sep = "-" * len(header)
    print("\n\n" + "!" * 60)
    print("                 FINAL EVALUATION SUMMARY")
    print("!" * 60)
    print(header)
    print(sep)
    for item in data:
        print(f"{item['Dataset']:<20} | {item['Model']:<12} | {item['PSNR']:>8} | {item['SSIM']:>8} | {item['LPIPS']:>8} | {item['NIQE']:>8} | {item['BRISQUE']:>8}")
        if item['Model'] == 'CIDNet_SSM':
            print(sep)
    print("!" * 60)


def make_comparison_image(images_dict, labels, header_h=40):
    """images_dict: ordered dict of {label: PIL.Image}, all same size"""
    imgs = list(images_dict.values())
    w, h = imgs[0].size
    n = len(imgs)
    combined = Image.new('RGB', (w * n, h + header_h), (255, 255, 255))
    for i, img in enumerate(imgs):
        combined.paste(img, (w * i, header_h))
    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    for i, label in enumerate(labels):
        draw.text((w * i + 10, 10), label, fill=(0, 0, 0), font=font)
    return combined


if __name__ == '__main__':
    parser = option()
    parser.add_argument('--input_image', type=str, default=None, help='Path to input image')
    parser.add_argument('--cidnet_ssm_path', type=str, default="weights/lolv2_syn/v5/20260122_235613_intensity_aug/epoch_500.pth", help='Path to the CIDNet SSM model weights')
    parser.add_argument('--output_dir', type=str, default='results/unpairedv1', help='Directory to save comparison images')
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
    parser.add_argument('--cidnet_path', type=str, default="Fediory/HVI-CIDNet-Generalization", help='Path to the base CIDNet model')
    parser.add_argument('--base_alpha_s', type=float, default=1.0, help='Base alpha_s parameter for CIDNet') 
    parser.add_argument('--base_alpha_i', type=float, default=1.0, help='Base alpha_i parameter for CIDNet')
    parser.add_argument('--alpha_rgb', type=float, default=1.0, help='RGB scaling factor')
    parser.add_argument('--use_GT_mean', type=bool, default=False, help='Use the mean of GT to rectify the output of the model')
    parser.add_argument('--unpaired_dataset_dir', type=str, default='./datasets/unpaired', help='Directory path containing unpaired datasets')
    parser.add_argument('--unpaired_dataset_names', type=str, nargs='+', default=['DICM', 'LIME', 'MEF', 'NPE'], help='List of unpaired dataset folder names to evaluate')
    args = parser.parse_args()

    alpha_combinations = [(args.base_alpha_s, args.base_alpha_i, args.alpha_rgb)]

    # ── Data loader setup ──────────────────────────────────────────────────
    if args.input_image is not None:
        input_img = Image.open(args.input_image)
        input_tensor = ToTensor()(input_img).unsqueeze(0)
        data_loader = [(input_tensor, torch.zeros_like(input_tensor))]
    else:
        training_data_loader, testing_data_loader = load_datasets(args)
        data_loader = testing_data_loader

    # ── Load CIDNet_SSM & register hook ───────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_net = CIDNet_SMM().cuda()
    eval_net.load_state_dict(
        torch.load(args.cidnet_ssm_path, map_location='cpu')['model_state_dict']
    )
    print(f"Loaded CIDNet_SSM checkpoint from {args.cidnet_ssm_path}")

    alpha_outputs = []
    hook = eval_net.alpha_predictor.register_forward_hook(
        lambda m, i, o: alpha_outputs.append(o)
    )

    results = eval(eval_net, data_loader, alpha_combinations)
    output_list, gt_list = results[alpha_combinations[0]]
    hook.remove()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Single-image mode ─────────────────────────────────────────────────
    if args.input_image is not None:
        Image.fromarray((output_list[0] * 255).astype(np.uint8)).save(
            os.path.join(args.output_dir, 'single_image_output.png')
        )
        print(f"Saved output to {args.output_dir}/single_image_output.png")

        if alpha_outputs:
            sf = alpha_outputs[0]  # shape: (1, 2, h, w)
            for ch, name in [(0, 'alpha_s'), (1, 'alpha_i')]:
                array_to_heatmap(sf[0, ch].cpu().numpy()).save(
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

        # Evaluate base CIDNet
        results_base = eval(cidnet_base, data_loader, alpha_combinations)
        output_base_list, _ = results_base[alpha_combinations[0]]

        # Paired metrics
        psnr_ssm, ssim_ssm, lpips_ssm = metrics(output_list, gt_list, use_GT_mean=args.use_GT_mean)
        psnr_base, ssim_base, lpips_base = metrics(output_base_list, gt_list, use_GT_mean=args.use_GT_mean)

        final_summary_data = [
            make_row('Paired (LOLv2)', 'Base CIDNet',
                     psnr=f"{psnr_base:.4f}", ssim=f"{ssim_base:.4f}", lpips=f"{lpips_base:.4f}"),
            make_row('Paired (LOLv2)', 'CIDNet_SSM',
                     psnr=f"{psnr_ssm:.4f}", ssim=f"{ssim_ssm:.4f}", lpips=f"{lpips_ssm:.4f}"),
        ]

        # ── Unpaired datasets ──────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("Unpaired Dataset Evaluation")
        print("=" * 60)

        total_imgs = total_niqe_ssm = total_brisque_ssm = total_niqe_base = total_brisque_base = 0

        for ds_name in args.unpaired_dataset_names:
            ds_path = os.path.join(args.unpaired_dataset_dir, ds_name)
            if not os.path.exists(ds_path):
                print(f"Skipping {ds_name}: path not found ({ds_path})")
                continue

            print(f"-- Evaluating {ds_name}...", end=" ", flush=True)
            ds_set = SICEDatasetFromFolderEval(ds_path, transform=ToTensor())
            ds_loader = DataLoader(dataset=ds_set, num_workers=1, batch_size=1, shuffle=False)

            outputs_ssm = eval(eval_net, ds_loader, alpha_combinations, unpaired=True)[alpha_combinations[0]][0]
            outputs_base = eval(cidnet_base, ds_loader, alpha_combinations, unpaired=True)[alpha_combinations[0]][0]

            niqe_ssm, brisque_ssm = metrics_no_ref(outputs_ssm)
            niqe_base, brisque_base = metrics_no_ref(outputs_base)
            print("Done")

            final_summary_data += [
                make_row(ds_name, 'Base CIDNet', niqe=f"{niqe_base:.4f}", brisque=f"{brisque_base:.4f}"),
                make_row(ds_name, 'CIDNet_SSM',  niqe=f"{niqe_ssm:.4f}",  brisque=f"{brisque_ssm:.4f}"),
            ]

            n = len(outputs_ssm)
            total_imgs += n
            total_niqe_ssm += niqe_ssm * n;   total_brisque_ssm += brisque_ssm * n
            total_niqe_base += niqe_base * n;  total_brisque_base += brisque_base * n

            # Save unpaired comparison images
            save_ds_dir = os.path.join(args.output_dir, ds_name)
            os.makedirs(save_ds_dir, exist_ok=True)

            ds_inputs = [
                (ds_set[i][0] if isinstance(ds_set[i], tuple) else ds_set[i]).numpy().transpose(1, 2, 0)
                for i in range(len(ds_set))
            ]
            for i, (out_ssm, out_base, in_np) in enumerate(zip(outputs_ssm, outputs_base, ds_inputs)):
                cmp = make_comparison_image(
                    {'Input': Image.fromarray((in_np * 255).astype(np.uint8)),
                     'Base CIDNet': Image.fromarray((out_base * 255).astype(np.uint8)),
                     'CIDNet_SSM': Image.fromarray((out_ssm * 255).astype(np.uint8))},
                    labels=["Input (No GT)", "Base CIDNet", "CIDNet_SSM"]
                )
                cmp.save(os.path.join(save_ds_dir, f'{ds_name}_{i+1:03d}_cmp.png'))

        # Unpaired average row
        if total_imgs > 0:
            final_summary_data += [
                make_row('AVERAGE (Unpaired)', 'Base CIDNet',
                         niqe=f"{total_niqe_base/total_imgs:.4f}", brisque=f"{total_brisque_base/total_imgs:.4f}"),
                make_row('AVERAGE (Unpaired)', 'CIDNet_SSM',
                         niqe=f"{total_niqe_ssm/total_imgs:.4f}",  brisque=f"{total_brisque_ssm/total_imgs:.4f}"),
            ]

        print_table(final_summary_data)

        # ── Save paired comparison images ──────────────────────────────────
        input_images = [batch[0].squeeze(0).numpy().transpose(1, 2, 0) for batch in testing_data_loader]

        dirs = ['input', 'cidnet', 'cidnet_ssm', 'gt', 'alpha_s', 'alpha_i', 'comparison']
        dir_paths = {name: os.path.join(args.output_dir, name) for name in dirs}
        for path in dir_paths.values():
            os.makedirs(path, exist_ok=True)

        cmp_labels = ["Input", "Base CIDNet", "CIDNet_SSM", "GT", "Alpha_s", "Alpha_i"]
        for idx, (out_np, base_np, gt_img, in_np, a_s, a_i) in enumerate(
            zip(output_list, output_base_list, gt_list, input_images, alpha_s_list, alpha_i_list)
        ):
            img_name = f'{idx+1:03d}.png'
            images = {
                'input':      Image.fromarray((in_np * 255).astype(np.uint8)),
                'cidnet':     Image.fromarray((base_np * 255).astype(np.uint8)),
                'cidnet_ssm': Image.fromarray((out_np * 255).astype(np.uint8)),
                'gt':         gt_img,
                'alpha_s':    array_to_heatmap(a_s),
                'alpha_i':    array_to_heatmap(a_i),
            }
            for name, img in images.items():
                img.save(os.path.join(dir_paths[name], img_name))

            make_comparison_image(images, cmp_labels).save(
                os.path.join(dir_paths['comparison'], f'comparison_{idx+1:03d}.png')
            )
            print(f"Saved images [{idx+1:03d}]: {', '.join(images.keys())}")

        print(f"\n✓ Saved {len(output_list)} images to:")
        for name, path in dir_paths.items():
            print(f"  - {name.capitalize()}: {path}")

