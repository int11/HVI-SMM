import os
import sys
import time
import random
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.options import option
from data.unpaired_dataset import UnpairedDataset
from data.data import transform1, transform2 # transform1 is for training (crop+augment)
from net.CIDNet_SMM import CIDNet_SMM
from net.EnlightenGAN import define_D
from loss.gan_losses import RaGANLoss, SelfFeaturePreservingLoss, GANLoss
from loss.zerodce_losses import SpatialConsistencyLoss, ExposureControlLoss, ColorConstancyLoss, IlluminationSmoothnessLoss
import scripts.dist as dist
from scripts.utils import Tee, checkpoint, compute_model_complexity, init_seed, build_generator_from_args, make_common_scheduler, plot_from_tfevents
from scripts.measure import metrics, metrics_no_ref
from scripts.eval import eval as eval_fn
import scripts.measure as measure_mod

def enlightengan_option():
    # option()이 반환한 parser에 EnlightenGAN 전용 인자를 추가
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

    # ---- EnlightenGAN 세부 설정 ----
    parser.add_argument('--patch_size',      type=int,   default=32,       help='지역 판별기용 패치 크기')
    parser.add_argument('--patch_d_num',     type=int,   default=5,        help='지역 판별기를 위한 패치 샘플링 개수')
    parser.add_argument('--vgg_loss_weight', type=float, default=1.0,      help='SFP (VGG) loss 가중치')
    parser.add_argument('--vgg_choose',     type=str,   default='relu5_1', help='VGG 특징 추출 레이어')
    parser.add_argument('--no_vgg_instance', action='store_true',          help='SFP loss에서 Instance Norm 사용 안 함')
    parser.add_argument('--no_ragan',       action='store_true',          help='RaGAN 대신 일반 GAN loss 사용')

    # ---- Physics Losses (Zero-DCE 기반, lambda=0이면 비활성) ----
    parser.add_argument('--lambda_spa',   type=float, default=0,
                        help='Spatial Consistency Loss 가중치')
    parser.add_argument('--lambda_exp',   type=float, default=0,
                        help='Exposure Control Loss 가중치')
    parser.add_argument('--lambda_col',   type=float, default=0,
                        help='Color Constancy Loss 가중치')
    parser.add_argument('--lambda_tv',    type=float, default=0,
                        help='Illumination Smoothness Loss (TV) 가중치')

    # Override default model_file for SMM
    parser.set_defaults(model_file='net/CIDNet_SMM.py')

    return parser

class EnlightenGANModel:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        # Generator: CIDNet_SMM
        self.netG = build_generator_from_args(args).to(device)
        
        # Discriminators
        self.netD_global = define_D(3, ndf=64, n_layers=3, norm='batch').to(device)
        self.netD_local = define_D(3, ndf=64, n_layers=3, norm='batch').to(device)
        
        # Losses
        if args.no_ragan:
            self.criterionGAN = GANLoss(use_lsgan=True).to(device)
        else:
            self.criterionGAN = RaGANLoss().to(device)
            
        self.criterionLocalGAN = GANLoss(use_lsgan=True).to(device) # Local은 기본적으로 LSGAN 사용
        
        self.criterionSFP = SelfFeaturePreservingLoss(
            layer_name=args.vgg_choose, 
            use_instance_norm=not args.no_vgg_instance
        ).to(device)
        
        # Optimizers (EnlightenGAN은 보통 beta1=0.5 사용)
        self.optimizer_G = optim.Adam(self.netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(
            list(self.netD_global.parameters()) + list(self.netD_local.parameters()),
            lr=args.lr, betas=(0.5, 0.999)
        )
        
        # Physics Losses (lambda > 0인 경우에만 인스턴스 생성)
        self.phys_losses = {}
        if args.lambda_spa > 0: self.phys_losses['spa'] = SpatialConsistencyLoss().to(device)
        if args.lambda_exp > 0: self.phys_losses['exp'] = ExposureControlLoss().to(device)
        if args.lambda_col > 0: self.phys_losses['col'] = ColorConstancyLoss().to(device)
        if args.lambda_tv  > 0: self.phys_losses['tv']  = IlluminationSmoothnessLoss().to(device)

        # Schedulers
        self.schedulers = [
            make_common_scheduler(self.optimizer_G, args),
            make_common_scheduler(self.optimizer_D, args)
        ]

    def set_input(self, real_low, real_high):
        self.real_low = real_low.to(self.device)
        self.real_high = real_high.to(self.device)

    def forward(self):
        # Generator forward
        output = self.netG(self.real_low)
        if isinstance(output, (tuple, list)):
            self.fake_high, self.fake_high_base = output[0], output[1]
        else:
            self.fake_high, self.fake_high_base = output, None
        
        # Random patches for local discriminator
        w = self.real_low.size(3)
        h = self.real_low.size(2)
        ps = self.args.patch_size
        num_patches = self.args.patch_d_num
        
        self.fake_patches = []
        self.real_high_patches = []
        self.real_low_patches = []
        
        for _ in range(num_patches):
            # Patch for fake_high and real_low (spatial correspondence)
            w_offset = random.randint(0, max(0, w - ps - 1))
            h_offset = random.randint(0, max(0, h - ps - 1))
            self.fake_patches.append(self.fake_high[:, :, h_offset:h_offset + ps, w_offset:w_offset + ps])
            self.real_low_patches.append(self.real_low[:, :, h_offset:h_offset + ps, w_offset:w_offset + ps])
            
            # Independent patch for real_high (unpaired, so no spatial correspondence)
            w_offset_high = random.randint(0, max(0, w - ps - 1))
            h_offset_high = random.randint(0, max(0, h - ps - 1))
            self.real_high_patches.append(self.real_high[:, :, h_offset_high:h_offset_high + ps, w_offset_high:w_offset_high + ps])
            
        self.fake_patches = torch.cat(self.fake_patches, dim=0)
        self.real_high_patches = torch.cat(self.real_high_patches, dim=0)
        self.real_low_patches = torch.cat(self.real_low_patches, dim=0)

    def backward_D(self):
        # Global Discriminator
        pred_real = self.netD_global(self.real_high)
        pred_fake = self.netD_global(self.fake_high.detach())
        
        if self.args.no_ragan:
            self.loss_D_global = (self.criterionGAN(pred_real, True) + self.criterionGAN(pred_fake, False)) / 2
        else:
            # RaGAN Loss for Global D
            self.loss_D_global = (self.criterionGAN(pred_real - torch.mean(pred_fake), True) +
                                 self.criterionGAN(pred_fake - torch.mean(pred_real), False)) / 2
        
        # Local Discriminator
        pred_real_patch = self.netD_local(self.real_high_patches)
        pred_fake_patch = self.netD_local(self.fake_patches.detach())
        self.loss_D_local = (self.criterionLocalGAN(pred_real_patch, True) +
                            self.criterionLocalGAN(pred_fake_patch, False)) / 2
        
        self.loss_D = self.loss_D_global + self.loss_D_local
        self.loss_D.backward()

    def backward_G(self):
        # Adversarial Loss (Global)
        pred_real = self.netD_global(self.real_high)
        pred_fake = self.netD_global(self.fake_high)
        
        if self.args.no_ragan:
            self.loss_G_GAN_global = self.criterionGAN(pred_fake, True)
        else:
            self.loss_G_GAN_global = (self.criterionGAN(pred_real - torch.mean(pred_fake), False) +
                                     self.criterionGAN(pred_fake - torch.mean(pred_real), True)) / 2
        
        # Adversarial Loss (Local)
        pred_fake_patch = self.netD_local(self.fake_patches)
        self.loss_G_GAN_local = self.criterionLocalGAN(pred_fake_patch, True)
        
        # SFP Loss (Global)
        self.loss_G_SFP_global = self.criterionSFP(self.fake_high, self.real_low)
        
        # SFP Loss (Local)
        self.loss_G_SFP_local = self.criterionSFP(self.fake_patches, self.real_low_patches)
        
        # Total G Loss
        self.loss_G = self.loss_G_GAN_global + self.loss_G_GAN_local + \
                      self.args.vgg_loss_weight * (self.loss_G_SFP_global + self.loss_G_SFP_local)
        
        if self.fake_high_base is not None:
            self.loss_G_SFP_base = self.criterionSFP(self.fake_high_base, self.real_low)
            self.loss_G += self.args.intermediate_weight * self.loss_G_SFP_base

        # Physics Losses (Zero-DCE 기반)
        if 'spa' in self.phys_losses:
            self.loss_phys_spa = self.phys_losses['spa'](self.real_low, self.fake_high)
            self.loss_G += self.args.lambda_spa * self.loss_phys_spa
        if 'exp' in self.phys_losses:
            self.loss_phys_exp = self.phys_losses['exp'](self.fake_high)
            self.loss_G += self.args.lambda_exp * self.loss_phys_exp
        if 'col' in self.phys_losses:
            self.loss_phys_col = self.phys_losses['col'](self.fake_high)
            self.loss_G += self.args.lambda_col * self.loss_phys_col
        if 'tv' in self.phys_losses:
            self.loss_phys_tv = self.phys_losses['tv'](self.fake_high)
            self.loss_G += self.args.lambda_tv * self.loss_phys_tv

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        
        # Update D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        
        # Update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            if scheduler is not None:
                scheduler.step()

    def get_current_losses(self):
        losses = {
            'G_Loss': self.loss_G.item(),
            'G_GAN_G': self.loss_G_GAN_global.item(),
            'G_GAN_L': self.loss_G_GAN_local.item(),
            'G_SFP_G': self.loss_G_SFP_global.item(),
            'D_Loss': self.loss_D.item(),
            'D_Global': self.loss_D_global.item(),
            'D_Local': self.loss_D_local.item(),
        }
        if hasattr(self, 'loss_phys_spa'): losses['Phys_SPA'] = self.loss_phys_spa.item()
        if hasattr(self, 'loss_phys_exp'): losses['Phys_EXP'] = self.loss_phys_exp.item()
        if hasattr(self, 'loss_phys_col'): losses['Phys_COL'] = self.loss_phys_col.item()
        if hasattr(self, 'loss_phys_tv'):  losses['Phys_TV']  = self.loss_phys_tv.item()
        return losses



def train(rank, args):
    if rank is not None:
        dist.init_distributed(rank)

    device = dist.get_device()
    now = time.strftime("%Y%m%d_%H%M%S")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(project_root, 'weights', f"enlightengan_{args.dataset}", now)
    
    if dist.is_main_process():
        os.makedirs(save_dir, exist_ok=True)
    
    with Tee(os.path.join(save_dir, 'train_log.txt')):
        init_seed(args.seed + (rank if rank is not None else 0))
        
        writer = None
        if dist.is_main_process():
            writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))
            print(f"Logging to {save_dir}")

        print(args)
        # Dataset (Unpaired)
        low_dir = args.data_low
        high_dir = args.data_high
        
        train_dataset = UnpairedDataset(low_dir, high_dir, transform=transform1(args.crop_size))
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.threads, pin_memory=True
        )
        
        if dist.is_dist_available_and_initialized():
            train_loader = dist.warp_loader(train_loader, True)

        # Evaluation Dataset (FiveK paired)
        from data.eval_sets import DatasetFromFolderEval
        val_dataset = DatasetFromFolderEval(args.data_val_low.replace('/input', ''), folder1='input', folder2='target', transform=transform2())
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

        # Model
        model = EnlightenGANModel(args, device)
        
        # Snapshots and iteration logic
        for epoch in range(args.start_epoch + 1, args.n_epochs + 1):
            if dist.is_dist_available_and_initialized():
                train_loader.sampler.set_epoch(epoch)
            
            epoch_start_time = time.time()
            total_losses = {}
            num_batches = 0
            
            for i, batch in enumerate(train_loader):
                real_low, real_high = batch[0], batch[1]
                model.set_input(real_low, real_high)
                model.optimize_parameters()
                
                losses = model.get_current_losses()
                for k, v in losses.items():
                    total_losses[k] = total_losses.get(k, 0) + v
                num_batches += 1
                
            # Epoch end
            epoch_time = time.time() - epoch_start_time
            model.update_learning_rate()
            
            if dist.is_main_process():
                avg_losses = {k: v / num_batches for k, v in total_losses.items()}
                metrics_str = " ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])
                print(f"===> Epoch {epoch} Complete. Time: {epoch_time:.2f}s | {metrics_str}")
                for k, v in avg_losses.items():
                    writer.add_scalar(f"Loss/{k}", v, epoch)
                writer.flush()
                plot_from_tfevents(save_dir)

                # Save checkpoint
                if epoch % args.snapshots == 0:
                    checkpoint(epoch, model.netG, model.optimizer_G, save_dir, filename=f"epoch_{epoch}_G.pth")
                    # Also save D if needed
                    torch.save({
                        'epoch': epoch,
                        'D_state_dict': {
                            'global': model.netD_global.state_dict(),
                            'local': model.netD_local.state_dict()
                        },
                        'optimizer_D_state_dict': model.optimizer_D.state_dict(),
                    }, os.path.join(save_dir, f"epoch_{epoch}_D.pth"))

                    # ---- Evaluation ----
                    print(f"===> Evaluating at Epoch {epoch}...")
                    alpha_combinations = [(1.0, 1.0, 1.0)] # Default alpha
                    eval_results = eval_fn(model.netG, val_loader, alpha_combinations)
                    output_list, gt_list = eval_results[alpha_combinations[0]]
                    
                    # Paired metrics (PSNR, SSIM, LPIPS)
                    avg_psnr, avg_ssim, avg_lpips = metrics(output_list, gt_list, use_gt_mean=False)
                    avg_psnr_gt, avg_ssim_gt, _ = metrics(output_list, gt_list, use_gt_mean=True)
                    
                    # No-reference metrics (NIQE, BRISQUE)
                    avg_niqe, avg_brisque = metrics_no_ref(output_list)
                    
                    print(f"===> Eval Results - PSNR: {avg_psnr:.4f} (GT-mean: {avg_psnr_gt:.4f}), SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")
                    print(f"===> Unsupervised Metrics - NIQE: {avg_niqe:.4f}, BRISQUE: {avg_brisque:.4f}")
                    
                    if writer:
                        writer.add_scalar("Metrics/PSNR", avg_psnr, epoch)
                        writer.add_scalar("Metrics/PSNR_GT", avg_psnr_gt, epoch)
                        writer.add_scalar("Metrics/SSIM", avg_ssim, epoch)
                        writer.add_scalar("Metrics/LPIPS", avg_lpips, epoch)
                        writer.add_scalar("Metrics/NIQE", avg_niqe, epoch)
                        writer.add_scalar("Metrics/BRISQUE", avg_brisque, epoch)
                    
                    model.netG.train() # Back to training mode
                    
                # Periodic visualization
                val_folder = args.val_folder
                if val_folder.startswith('./'):
                    val_folder = os.path.join(project_root, val_folder[2:])
                val_save_path = os.path.join(val_folder, 'enlightengan')
                os.makedirs(val_save_path, exist_ok=True)
                
                input_vis = model.real_low[0].cpu().clamp(0, 1)
                output_vis = model.fake_high[0].cpu().clamp(0, 1)
                # Concatenate horizontally
                combined_vis = torch.cat([input_vis, output_vis], dim=2)
                
                res_img = transforms.ToPILImage()(combined_vis)
                res_img.save(os.path.join(val_save_path, f"epoch_{epoch}.png"))

        if writer:
            writer.close()

if __name__ == '__main__':
    args = enlightengan_option().parse_args()
    
    if args.gpu_mode:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    
    world_size = torch.cuda.device_count()
    if world_size > 1:
        import torch.multiprocessing as mp
        mp.spawn(train, args=(args,), nprocs=world_size, join=True)
    else:
        train(None, args)
