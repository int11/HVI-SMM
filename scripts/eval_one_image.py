import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data import *
from loss.losses import *
from scripts.measure import metrics
import scripts.dist as dist
from scripts.options import option
import safetensors.torch as sf
from huggingface_hub import hf_hub_download
from net.CIDNet_SMM import CIDNet_SMM
from net.CIDNet import CIDNet
from net.CIDNet_fix import CIDNet_fix
from torchvision.transforms import ToPILImage
from PIL import Image
import torch
import numpy as np

def eval_one_image(model, input_image, alpha_predict=True, base_alpha_s=1.0, base_alpha_i=1.0, alpha_rgb=1.0):
    torch.set_grad_enabled(False)
    model = dist.de_parallel(model)
    model.eval()
    if isinstance(model, CIDNet_SMM):
        model.set_alpha_predict(alpha_predict)
        model.set_base_alpha(base_alpha_s, base_alpha_i, alpha_rgb)
    elif isinstance(model, (CIDNet, CIDNet_fix)):
        model.set_base_alpha(base_alpha_s, base_alpha_i, alpha_rgb)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = torch.from_numpy(input_image.transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255.0
    with torch.no_grad():
        if isinstance(model, CIDNet_SMM):
            output, _ = model(input_tensor)
        elif isinstance(model, (CIDNet, CIDNet_fix)):
            output = model(input_tensor)
    output = torch.clamp(output, 0, 1).to(device)
    output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    torch.set_grad_enabled(True)
    return output_np

def load_cidnet_base_model(model_path, device):
    print(f"Loading CIDNet model from: {model_path}")
    model_file = hf_hub_download(
        repo_id=model_path,
        filename="model.safetensors",
        repo_type="model"
    )
    print(f"CIDNet model downloaded from: {model_file}")
    model = CIDNet()
    state_dict = sf.load_file(model_file)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate one image using CIDNet or CIDNet_SSM')
    parser.add_argument('--input_image', type=str, required=True, help='Path to input image')
    parser.add_argument('--weight_path', type=str, default=None, help='Path to the pre-trained model weights (for CIDNet_SSM)')
    parser.add_argument('--cidnet_model', type=str, default=None, help='CIDNet model name or path from Hugging Face')
    parser.add_argument('--output_path', type=str, default='output.png', help='Path to save output image')
    parser.add_argument('--base_alpha_s', type=float, default=1.0, help='Base alpha_s parameter for CIDNet')
    parser.add_argument('--base_alpha_i', type=float, default=1.0, help='Base alpha_i parameter for CIDNet')
    parser.add_argument('--alpha_rgb', type=float, default=1.0, help='RGB scaling factor')
    parser.add_argument('--resize', type=int, nargs=2, default=[384, 512], help='Resize size for input image, e.g. --resize 384 384')
    parser.add_argument('--use_ssm', action='store_true', help='Use CIDNet_SSM model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_img = Image.open(args.input_image).convert('RGB')
    input_img = input_img.resize(tuple(args.resize), Image.BICUBIC)
    input_np = np.array(input_img)

    if args.use_ssm:
        assert args.weight_path is not None, 'weight_path required for CIDNet_SSM.'
        model = CIDNet_SMM().to(device)
        checkpoint_data = torch.load(args.weight_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        print(f"Loaded CIDNet_SSM checkpoint from {args.weight_path}")
        output_np = eval_one_image(model, input_np, alpha_predict=True, base_alpha_s=args.base_alpha_s, base_alpha_i=args.base_alpha_i, alpha_rgb=args.alpha_rgb)
    else:
        assert args.cidnet_model is not None, 'cidnet_model required for base CIDNet.'
        model = load_cidnet_base_model(args.cidnet_model, device)
        print(f"Loaded base CIDNet model from {args.cidnet_model}")
        output_np = eval_one_image(model, input_np, alpha_predict=False, base_alpha_s=args.base_alpha_s, base_alpha_i=args.base_alpha_i, alpha_rgb=args.alpha_rgb)

    output_img = Image.fromarray((output_np * 255).astype(np.uint8))
    # 원본 이미지 로드 및 리사이즈
    original_img = Image.open(args.input_image).convert('RGB')
    orig_w, orig_h = original_img.size
    # 모델 결과를 원본 사이즈로 리사이즈
    output_img_resized = output_img.resize((orig_w, orig_h), Image.BICUBIC)
    resize_w, resize_h = tuple(args.resize)
    # 라벨 영역 높이
    label_height = 50
    # 비교 이미지: [원본 | 리사이즈 | 결과 | 결과(원본사이즈)] 가로로 나란히, 아래에 라벨
    total_width = orig_w + resize_w + resize_w + orig_w
    total_height = max(orig_h, resize_h) + label_height
    comparison_img = Image.new('RGB', (total_width, total_height), (255,255,255))
    comparison_img.paste(original_img, (0, 0))
    comparison_img.paste(input_img, (orig_w, 0))
    comparison_img.paste(output_img, (orig_w + resize_w, 0))
    comparison_img.paste(output_img_resized, (orig_w + resize_w + resize_w, 0))
    # 한글 라벨 추가
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(comparison_img)
    try:
        font = ImageFont.truetype("NanumGothic.ttf", 28)
    except:
        font = ImageFont.load_default()
    label_y = max(orig_h, resize_h) + 10
    draw.text((orig_w//2 - 60, label_y), "Original", fill="black", font=font)
    draw.text((orig_w + resize_w//2 - 60, label_y), "Resized", fill="black", font=font)
    draw.text((orig_w + resize_w + resize_w//2 - 60, label_y), "Model Output", fill="black", font=font)
    draw.text((orig_w + resize_w + resize_w + orig_w//2 - 100, label_y), "Model Output (Original Size)", fill="black", font=font)
    comparison_img.save(args.output_path)
    print(f"Saved comparison image to {args.output_path}")
