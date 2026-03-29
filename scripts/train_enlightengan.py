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
import scripts.dist as dist
from scripts.utils import Tee, checkpoint, compute_model_complexity, init_seed, build_generator_from_args, make_common_scheduler
from scripts.measure import metrics
from scripts.eval import eval as eval_fn

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
            w_offset = random.randint(0, max(0, w - ps - 1))
            h_offset = random.randint(0, max(0, h - ps - 1))
            self.fake_patches.append(self.fake_high[:, :, h_offset:h_offset + ps, w_offset:w_offset + ps])
            self.real_high_patches.append(self.real_high[:, :, h_offset:h_offset + ps, w_offset:w_offset + ps])
            self.real_low_patches.append(self.real_low[:, :, h_offset:h_offset + ps, w_offset:w_offset + ps])
            
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
        return {
            'G_Loss': self.loss_G.item(),
            'G_GAN_G': self.loss_G_GAN_global.item(),
            'G_GAN_L': self.loss_G_GAN_local.item(),
            'G_SFP_G': self.loss_G_SFP_global.item(),
            'D_Loss': self.loss_D.item(),
            'D_Global': self.loss_D_global.item(),
            'D_Local': self.loss_D_local.item(),
        }



def train(rank, args):
    if rank is not None:
        dist.init_distributed(rank)

    device = dist.get_device()
    now = time.strftime("%Y%m%d_%H%M%S")
    save_dir = f"./weights/enlightengan_{args.dataset}/{now}"
    
    with Tee(os.path.join(save_dir, 'train_log.txt')):
        init_seed(args.seed + (rank if rank is not None else 0))
        
        writer = None
        if dist.is_main_process():
            writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))
            print(f"Logging to {save_dir}")

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
                    
                    # Periodic visualization
                    if not os.path.exists(args.val_folder + 'enlightengan'):
                        os.makedirs(args.val_folder + 'enlightengan')
                    res_img = transforms.ToPILImage()(model.fake_high[0].cpu().clamp(0, 1))
                    res_img.save(f"{args.val_folder}enlightengan/epoch_{epoch}.png")

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
