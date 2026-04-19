"""
FoFA (Foundation Feature-space Alignment) 비지도 GAN 학습 스크립트.

논문: "Improving Visual and Downstream Performance of Low-Light Enhancer
       with Vision Foundation Models Collaboration" (CVPR 2025)

학습 데이터:
  - Low-light  : ExDark (--data_low)
  - Normal-light: COCO 2017 train (--data_high)
  - Eval        : paired test set (--data_val_low / --data_val_high)

주요 설계:
  - options.py 수정 없음: 공통 인자는 parents=[option()] 로 상속
  - FoFA 전용 인자는 이 파일 내부 argparse에만 정의
  - DDP 패턴은 scripts/dist.py와 동일하게 사용
  - G / D 이중 optimizer + scheduler
  - 체크포인트: scripts/utils.py의 fofa_checkpoint / fofa_load_checkpoint 사용

실행 예:
  # single GPU
  python scripts/train_fofa.py \\
      --model_file net/CIDNet_fix.py \\
      --data_low ./datasets/ExDark \\
      --data_high ./datasets/COCO/train2017 \\
      --data_val_low ./datasets/LOL-v2/Synthetic/Test/Low \\
      --data_val_high ./datasets/LOL-v2/Synthetic/Test/Normal \\
      --nEpochs 50 --batchSize 8 --cropSize 448

  # multi GPU (자동 탐지)
  python scripts/train_fofa.py --data_low ... --data_high ...
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import inspect
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, ToTensor, Resize
from torchvision import utils as vutils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from scripts.options import option          # 공통 인자 확장용 (option() 반환 parser에 add_argument)
from scripts.eval import eval as eval_model
from scripts.measure import metrics
from scripts.utils import Tee, fofa_checkpoint, fofa_load_checkpoint, compute_model_complexity, init_seed, build_generator_from_args, make_common_scheduler, plot_from_tfevents
from data.scheduler import *                # CosineAnnealingRestartLR, GradualWarmupScheduler 등
import scripts.dist as dist

from data.unpaired_dataset import UnpairedDataset
from data.eval_sets import DatasetFromFolderEval

from net.fofa_modules import FoundationEncoder, FeatureProjector, FoFADiscriminator
from loss.losses import (FoFADiscriminatorLoss, SpatialConsistencyLoss, 
                         ExposureControlLoss, ColorConstancyLoss, IlluminationSmoothnessLoss)


# ---------------------------------------------------------------------------
# Argument parser (FoFA 전용 인자 + 공통 인자 상속)
# ---------------------------------------------------------------------------

def fofa_option():
    # option()이 반환한 parser에 FoFA 전용 인자를 추가 (options.py 수정 없음)
    parser = option()

    # ---- Unpaired 데이터 경로 ----
    parser.add_argument('--data_low',      type=str, default='./datasets/ExDark', required=False,
                        help='저조도 학습 이미지 디렉토리 (예: ExDark, 기본값: ./datasets/ExDark)')
    parser.add_argument('--data_high',     type=str, default='./datasets/coco/train2017', required=False,
                        help='정상조도 학습 이미지 디렉토리 (예: COCO train2017, 기본값: ./datasets/coco/train2017)')
    parser.add_argument('--data_val_low',  type=str, default='./datasets/FiveK/test/input',
                        help='평가용 저조도 이미지 디렉토리')
    parser.add_argument('--data_val_high', type=str, default='./datasets/FiveK/test/target',
                        help='평가용 정상조도 이미지 디렉토리')

    # ---- FoFA 손실 가중치 ----
    parser.add_argument('--lambda_adv',   type=float, default=1.0,
                        help='Generator adversarial loss 가중치')
    parser.add_argument('--lambda_align', type=float, default=1.0,
                        help='D feature-foundation alignment loss 가중치 (Eq.8)')
    
    # ---- Baseline Physics Losses (Zero-DCE/SCI) ----
    parser.add_argument('--lambda_spa',   type=float, default=0,
                        help='Spatial Consistency Loss 가중치')
    parser.add_argument('--lambda_exp',   type=float, default=0,
                        help='Exposure Control Loss 가중치')
    parser.add_argument('--lambda_col',   type=float, default=0,
                        help='Color Constancy Loss 가중치')
    parser.add_argument('--lambda_tv',    type=float, default=0,
                        help='Illumination Smoothness Loss (TV) 가중치')

    # ---- Discriminator / Foundation 설정 ----
    parser.add_argument('--disc_lr',          type=float, default=1e-4,
                        help='Discriminator learning rate')
    parser.add_argument('--n_disc_steps',     type=int,   default=1,
                        help='G 1스텝당 D 업데이트 횟수')
    parser.add_argument('--fofa_proj_channels', type=int, default=256,
                        help='FeatureProjector 통일 채널 수')
    parser.add_argument('--foundation_models', type=str, default='clip,sam,resnet',
                        help='사용할 foundation 모델 (콤마 구분, 예: clip,sam,sam2,resnet)')

    # ---- Resume ----
    parser.add_argument('--resume_path', type=str, default='',
                        help='재개할 FoFA checkpoint 경로 (fofa_checkpoint 포맷)')
    parser.add_argument('--save_dir',    type=str, default='',
                        help='저장 디렉토리 override (빈 문자열이면 자동 생성)')

    return parser


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------

def transform_train(crop_size: int):
    return Compose([
        Resize((crop_size, crop_size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ])


class CropToMultiple:
    def __init__(self, multiple=16):
        self.multiple = multiple
    def __call__(self, img):
        # img is PIL Image
        w, h = img.size
        new_w = (w // self.multiple) * self.multiple
        new_h = (h // self.multiple) * self.multiple
        if new_w == w and new_h == h:
            return img
        return img.crop((0, 0, new_w, new_h))

def transform_eval():
    return Compose([CropToMultiple(16), ToTensor()])


def build_train_loader(args) -> DataLoader:
    dataset = UnpairedDataset(
        low_dir=args.data_low,
        high_dir=args.data_high,
        transform=transform_train(args.cropSize),
    )
    return DataLoader(
        dataset,
        batch_size=args.batchSize,
        shuffle=args.shuffle,
        num_workers=args.threads,
        pin_memory=True,
        drop_last=True,
    )


def build_eval_loader(args):
    """평가 데이터 로더. val 경로가 없으면 None 반환."""
    if not args.data_val_low or not args.data_val_high:
        return None
    # DatasetFromFolderEval은 두 폴더를 직접 받는 구조가 아니므로
    # 상위 디렉토리 + 서브폴더명으로 구성
    val_root = os.path.commonpath([args.data_val_low, args.data_val_high])
    low_name  = os.path.relpath(args.data_val_low,  val_root)
    high_name = os.path.relpath(args.data_val_high, val_root)
    dataset = DatasetFromFolderEval(
        val_root,
        folder1=low_name,
        folder2=high_name,
        transform=transform_eval(),
    )
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)


# ---------------------------------------------------------------------------
# 모델 / 모듈 초기화
# ---------------------------------------------------------------------------

def build_fofa_modules(args, device):
    """FoundationEncoder / FeatureProjector / FoFADiscriminator 초기화."""
    use_models = [m.strip() for m in args.foundation_models.split(',')]
    
    # 객체 지향적으로 리팩토링된 FoundationEncoder 호출
    encoder = FoundationEncoder(
        model_names=use_models,
        input_size=args.cropSize,
    ).to(device)

    projector = FeatureProjector(
        in_channels_dict=encoder.out_channels,
        proj_channels=args.fofa_proj_channels,
    ).to(device)

    # 파운데이션 모델의 개수를 리스트 길이로 동적 계산
    n_models = len(use_models)
    
    foundation_channels = args.fofa_proj_channels * n_models

    discriminator = FoFADiscriminator(
        in_channels=3,
        foundation_channels=foundation_channels,
        ndf=64,
        n_layers=3,
    ).to(device)

    return encoder, projector, discriminator


# ---------------------------------------------------------------------------
# 학습 스텝
# ---------------------------------------------------------------------------

def discriminator_step(im_low, im_high,
                       generator, discriminator, encoder, projector,
                       loss_fn, optimizer_d, args, device):
    """
    D 업데이트 (Eq.8):
      real  = D(im_high, f_high)
      fake  = D(G(im_low).detach(), f_fake)
    """
    generator.eval()
    discriminator.train()

    with torch.no_grad():
        gen_out = generator(im_low)
        fake_img = gen_out[0] if isinstance(gen_out, tuple) else gen_out

    # foundation features
    feat_dict_real = encoder(im_high)
    feat_dict_fake = encoder(fake_img)

    target_hw = (im_high.shape[2] // 8, im_high.shape[3] // 8)
    f_real, _ = projector(feat_dict_real, target_hw)
    f_fake, _ = projector(feat_dict_fake, target_hw)

    logit_real, feat_real, f_real_proj = discriminator(im_high, f_real)
    logit_fake, feat_fake, f_fake_proj = discriminator(fake_img.detach(), f_fake)

    loss_d = loss_fn.loss_D(
        logit_real, logit_fake,
        feat_real, f_real_proj,
        feat_fake, f_fake_proj,
    )

    optimizer_d.zero_grad()
    loss_d.backward()
    if args.grad_clip:
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
    optimizer_d.step()

    return loss_d.item()


def generator_step(im_low, im_high,
                   generator, discriminator, encoder, projector,
                   loss_fn, phys_losses, optimizer_g, args, device):
    """
    G 업데이트 (Eq.9 + optional physical baseline losses):
      adv   = -log D(G(low))
      phys  = L_spa + L_exp + L_col + L_tv (Zero-DCE 기반)
    """
    generator.train()
    discriminator.eval()

    gen_out = generator(im_low)
    fake_img = gen_out[0] if isinstance(gen_out, tuple) else gen_out

    feat_dict_fake = encoder(fake_img)
    target_hw = (fake_img.shape[2] // 8, fake_img.shape[3] // 8)
    f_fake, _ = projector(feat_dict_fake, target_hw)

    logit_fake, _, _ = discriminator(fake_img, f_fake)

    loss_adv = loss_fn.loss_G(logit_fake)
    loss_g = args.lambda_adv * loss_adv

    # Physics Baseline Losses (Zero-DCE)
    if 'spa' in phys_losses and args.lambda_spa > 0:
        loss_spa = phys_losses['spa'](im_low, fake_img)
        loss_g += args.lambda_spa * loss_spa
    if 'exp' in phys_losses and args.lambda_exp > 0:
        loss_exp = phys_losses['exp'](fake_img)
        loss_g += args.lambda_exp * loss_exp
    if 'col' in phys_losses and args.lambda_col > 0:
        loss_col = phys_losses['col'](fake_img)
        loss_g += args.lambda_col * loss_col
    if 'tv' in phys_losses and args.lambda_tv > 0:
        loss_tv = phys_losses['tv'](fake_img)
        loss_g += args.lambda_tv * loss_tv

    optimizer_g.zero_grad()
    loss_g.backward()
    if args.grad_clip:
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 0.01)
    optimizer_g.step()

    return loss_g.item(), fake_img.detach()


# ---------------------------------------------------------------------------
# 1 에폭 학습
# ---------------------------------------------------------------------------

def train_one_epoch(generator, discriminator, encoder, projector,
                    train_loader, optimizer_g, optimizer_d,
                    loss_fn, phys_losses, args, device):
    import time
    start = time.time()

    total_loss_g = total_loss_d = 0.0
    last_fake = None

    for batch_idx, batch in enumerate(train_loader, 1):

        im_low  = batch[0].to(device)
        im_high = batch[1].to(device)

        # --- 데이터 min/max/mean/std 체크 ---
        if batch_idx % 100 == 0 or batch_idx == 1:
            def stat(x):
                return f"min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}, std={x.std().item():.4f}"
            print(f"[DEBUG] im_low:  {stat(im_low)}")
            print(f"[DEBUG] im_high: {stat(im_high)}")

        # D 업데이트 (n_disc_steps 만큼)
        for _ in range(args.n_disc_steps):
            ld = discriminator_step(
                im_low, im_high,
                generator, discriminator, encoder, projector,
                loss_fn, optimizer_d, args, device
            )
            # Loss NaN/inf 체크
            if not torch.isfinite(torch.tensor(ld)):
                print(f"[ERROR] Discriminator loss is not finite: {ld}")
        total_loss_d += ld

        # G 업데이트
        lg, last_fake = generator_step(
            im_low, im_high,
            generator, discriminator, encoder, projector,
            loss_fn, phys_losses, optimizer_g, args, device
        )
        # Loss NaN/inf 체크
        if not torch.isfinite(torch.tensor(lg)):
            print(f"[ERROR] Generator loss is not finite: {lg}")
        total_loss_g += lg

        # --- Generator 출력 min/max/mean/std 체크 ---
        if batch_idx % 100 == 0 or batch_idx == 1:
            if isinstance(last_fake, torch.Tensor):
                print(f"[DEBUG] fake_img: {stat(last_fake)}")
            else:
                print(f"[DEBUG] fake_img: type={type(last_fake)}")

        # --- 저장 및 추가 디버깅 ---
        if batch_idx % 100 == 0 or batch_idx == len(train_loader):
            print(f"Batch {batch_idx}/{len(train_loader)} | Loss G: {lg:.4f} D: {ld:.4f}")
            if dist.is_main_process():
                sample_dir = os.path.join(args.val_folder, 'fofa_training')
                os.makedirs(sample_dir, exist_ok=True)
                transforms.ToPILImage()(im_low[0].detach().cpu()).save(
                    os.path.join(sample_dir, 'input_low.png'))
                transforms.ToPILImage()(im_high[0].detach().cpu()).save(
                    os.path.join(sample_dir, 'input_high.png'))
                transforms.ToPILImage()(last_fake[0].detach().cpu()).save(
                    os.path.join(sample_dir, 'fake.png'))

        # --- NaN/inf 체크 (출력) ---
        if isinstance(last_fake, torch.Tensor):
            if not torch.isfinite(last_fake).all():
                print("[ERROR] Generator output contains NaN or Inf values!")

    n = len(train_loader)
    return total_loss_g / n, total_loss_d / n, time.time() - start, last_fake


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------

def run_eval(generator, eval_loader, args):
    if eval_loader is None:
        return None, None, None
    alpha_combos = [(1.0, 1.0, 1.0)]
    results = eval_model(generator, eval_loader, alpha_combos)
    out_list, gt_list = results[(1.0, 1.0, 1.0)]
    avg_psnr, avg_ssim, avg_lpips = metrics(out_list, gt_list, use_GT_mean=True)
    return avg_psnr, avg_ssim, avg_lpips


# ---------------------------------------------------------------------------
# 메인 학습 함수
# ---------------------------------------------------------------------------

def train(rank, args):
    if rank is not None:
        dist.init_distributed(rank)

    device = dist.get_device()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = args.save_dir if args.save_dir else f"./weights/fofa/{now}"

    with Tee(os.path.join(save_dir, '1log.txt')):
        init_seed(args.seed + (rank if rank is not None else 0))

        writer = None
        if dist.is_main_process():
            writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))
            print(f"TensorBoard: {save_dir}/tensorboard")
        print(args)

        # ---- 데이터 로더 ----
        train_loader = build_train_loader(args)
        eval_loader  = build_eval_loader(args)

        # ---- 모델 ----
        print('===> Building generator')
        generator = build_generator_from_args(args).to(device)

        if dist.is_main_process():
            flops, params = compute_model_complexity(
                generator, input_size=(1, 3, args.cropSize, args.cropSize))
            print(f"Generator FLOPs: {flops}, Params: {params}")

        print('===> Building FoFA modules (Encoder / Projector / Discriminator)')
        encoder, projector, discriminator = build_fofa_modules(args, device)

        # ---- 손실 함수 ----
        loss_fn = FoFADiscriminatorLoss(lambda_feat=args.lambda_align).to(device)
        
        # Instantiate Physical Baseline Losses
        phys_losses = {}
        if args.lambda_spa > 0: phys_losses['spa'] = SpatialConsistencyLoss().to(device)
        if args.lambda_exp > 0: phys_losses['exp'] = ExposureControlLoss().to(device)
        if args.lambda_col > 0: phys_losses['col'] = ColorConstancyLoss().to(device)
        if args.lambda_tv  > 0: phys_losses['tv']  = IlluminationSmoothnessLoss().to(device)

        # ---- Optimizer ----
        optimizer_g = optim.Adam(
            list(generator.parameters()) + list(projector.parameters()),
            lr=args.lr
        )
        optimizer_d = optim.Adam(
            discriminator.parameters(),
            lr=args.disc_lr
        )

        # ---- Scheduler ----
        scheduler_g = make_common_scheduler(optimizer_g, args)
        scheduler_d = make_common_scheduler(optimizer_d, args)

        # ---- Resume ----
        start_epoch = 0
        if args.resume_path:
            start_epoch = fofa_load_checkpoint(
                args.resume_path,
                generator, discriminator,
                optimizer_g, optimizer_d,
                scheduler_g, scheduler_d,
                device=device,
            )
        elif args.start_epoch > 0:
            pth = f"./weights/fofa/epoch_{args.start_epoch}.pth"
            start_epoch = fofa_load_checkpoint(
                pth, generator, discriminator,
                optimizer_g, optimizer_d,
                scheduler_g, scheduler_d,
                device=device,
            )

        # ---- DDP wrapping ----
        if dist.is_dist_available_and_initialized():
            train_loader  = dist.warp_loader(train_loader, args.shuffle)
            generator     = dist.warp_model(generator,     sync_bn=True,  find_unused_parameters=False)
            discriminator = dist.warp_model(discriminator, sync_bn=False, find_unused_parameters=False)
            projector     = dist.warp_model(projector,     sync_bn=False, find_unused_parameters=False)
            # encoder는 freeze이므로 DDP 불필요하지만 multi-GPU 시 device 동기화용으로 wrapping
            encoder       = dist.warp_model(encoder,       sync_bn=False, find_unused_parameters=True)

        # ---- 학습 루프 ----
        for epoch in range(start_epoch + 1, args.nEpochs + start_epoch + 1):
            if dist.is_dist_available_and_initialized():
                train_loader.sampler.set_epoch(epoch)

            loss_g, loss_d, elapsed, last_fake = train_one_epoch(
                generator, discriminator, encoder, projector,
                train_loader, optimizer_g, optimizer_d,
                loss_fn, phys_losses, args, device
            )

            if scheduler_g is not None:
                scheduler_g.step()
            if scheduler_d is not None:
                scheduler_d.step()

            print(
                f"===> Epoch[{epoch}] "
                f"loss_G: {loss_g:.4f} | loss_D: {loss_d:.4f} | "
                f"lr_G: {optimizer_g.param_groups[0]['lr']:.2e} | "
                f"lr_D: {optimizer_d.param_groups[0]['lr']:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            if writer is not None:
                writer.add_scalar('Loss/G', loss_g, epoch)
                writer.add_scalar('Loss/D', loss_d, epoch)
                writer.add_scalar('LR/G',   optimizer_g.param_groups[0]['lr'], epoch)
                writer.add_scalar('LR/D',   optimizer_d.param_groups[0]['lr'], epoch)
                writer.flush()
                if dist.is_main_process():
                    plot_from_tfevents(save_dir)

            # Checkpoint + Eval
            if epoch % args.snapshots == 0 and dist.is_main_process():
                fofa_checkpoint(
                    epoch, generator, discriminator,
                    optimizer_g, optimizer_d,
                    scheduler_g, scheduler_d,
                    save_dir
                )
                if eval_loader is not None:
                    psnr, ssim, lpips = run_eval(generator, eval_loader, args)
                    print(f"===> Eval Epoch[{epoch}]: PSNR={psnr:.4f} SSIM={ssim:.4f} LPIPS={lpips:.4f}")
                    if writer is not None:
                        writer.add_scalar('Eval/PSNR',  psnr,  epoch)
                        writer.add_scalar('Eval/SSIM',  ssim,  epoch)
                        writer.add_scalar('Eval/LPIPS', lpips, epoch)

            torch.cuda.empty_cache()

        if writer is not None:
            writer.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = fofa_option().parse_args()

    if not args.gpu_mode:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

    world_size = torch.cuda.device_count()
    print(f"Detected {world_size} GPU(s)")

    if world_size > 1:
        import torch.multiprocessing as mp
        mp.spawn(train, args=(args,), nprocs=world_size, join=True)
    else:
        train(None, args)
