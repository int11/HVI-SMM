import os
import sys
import torch
import random
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from net.CIDNet import CIDNet
from net.CIDNet_SSM import CIDNet as CIDNet_sam
from data.options import option, load_datasets
from measure import metrics
from eval import eval
from data.data import *
from loss.losses import *
from data.scheduler import *
from tqdm import tqdm
from datetime import datetime
import glob
import dist


class Tee:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, 'a')
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self  # Redirect stdout to this instance
        print(f"===== Logging session started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        return self

    def write(self, obj):
        self.file.write(obj)
        self.file.flush()
        self.stdout.write(obj)
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"===== Logging session ended {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        sys.stdout = self.stdout  # Restore original stdout
        self.file.close()


def train_one_epoch(model, optimizer, training_data_loader, args, L1_loss, P_loss, E_loss, D_loss):
    model.train()
    total_loss = 0  # 전체 에폭의 손실 합계
    total_batches = 0  # 전체 배치 수
    torch.autograd.set_detect_anomaly(args.grad_detect)
    
    for batch_idx, batch in enumerate(training_data_loader, 1):
        im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3]

        # Use rank-based device assignment for distributed training
        device = dist.get_device()
    
        im1 = im1.to(device)
        im2 = im2.to(device)
        
        # use random gamma function (enhancement curve) to improve generalization
        if args.gamma:
            gamma = random.randint(args.start_gamma,args.end_gamma) / 100.0
            output_rgb = model(im1 ** gamma)  
        else:
            output_rgb = model(im1)  
            
        gt_rgb = im2
        
        # Get RGB_to_HVI method from the actual model (handle DDP wrapper)
        rgb_to_hvi_fn = model.module.RGB_to_HVI if hasattr(model, 'module') else model.RGB_to_HVI
        
        output_hvi = rgb_to_hvi_fn(output_rgb)
        gt_hvi = rgb_to_hvi_fn(gt_rgb)
        loss_hvi = L1_loss(output_hvi, gt_hvi) + D_loss(output_hvi, gt_hvi) + E_loss(output_hvi, gt_hvi) + args.P_weight * P_loss(output_hvi, gt_hvi)[0]
        loss_rgb = L1_loss(output_rgb, gt_rgb) + D_loss(output_rgb, gt_rgb) + E_loss(output_rgb, gt_rgb) + args.P_weight * P_loss(output_rgb, gt_rgb)[0]
        loss = loss_rgb + args.HVI_weight * loss_hvi
        
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 손실과 배치 수 누적
        total_loss += loss.item()
        total_batches += 1
        
    # 에폭 완료 후 샘플 이미지 저장 (마지막 배치의 결과)
    output_img = transforms.ToPILImage()((output_rgb)[0].squeeze(0))
    gt_img = transforms.ToPILImage()((gt_rgb)[0].squeeze(0))
    if not os.path.exists(args.val_folder+'training'):          
        os.mkdir(args.val_folder+'training') 
    output_img.save(args.val_folder+'training/test.png')
    gt_img.save(args.val_folder+'training/gt.png')
    
    return total_loss, total_batches
                

def checkpoint(epoch, model, optimizer, path):
    os.makedirs(path, exist_ok=True)
    model_out_path = os.path.join(path, f"epoch_{epoch}.pth")
    
    # Save model and optimizer states with epoch info
    # Use de_parallel to handle distributed models
    model_state = dist.de_parallel(model).state_dict() if dist.is_dist_available_and_initialized() else model.state_dict()
    
    checkpoint_dict = {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint_dict, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    

def build_model():
    print('===> Building model ')
    model = CIDNet()
    model = model.to(dist.get_device())
    return model


def make_scheduler(optimizer, args):
    # Calculate last_epoch for resumed training
    cosine_last_epoch = -1 if args.start_epoch == 0 else args.start_epoch - 1 - args.warmup_epochs
    
    if args.cos_restart_cyclic:
        # CosineAnnealingRestartCyclicLR scheduler
        periods = [(args.nEpochs//4)-args.warmup_epochs, (args.nEpochs*3)//4] if args.start_warmup else [args.nEpochs//4, (args.nEpochs*3)//4]
        scheduler_step = CosineAnnealingRestartCyclicLR(
            optimizer=optimizer, 
            periods=periods, 
            restart_weights=[1,1], 
            eta_mins=[0.0002,0.0000001],
            last_epoch=cosine_last_epoch
        )
        
    elif args.cos_restart:
        # CosineAnnealingRestartLR scheduler
        periods = [args.nEpochs - args.warmup_epochs] if args.start_warmup else [args.nEpochs]
        scheduler_step = CosineAnnealingRestartLR(
            optimizer=optimizer, 
            periods=periods, 
            restart_weights=[1], 
            eta_min=1e-7,
            last_epoch=cosine_last_epoch
        )
        
    else:
        raise Exception("should choose a scheduler")
    
    # Create main scheduler (with or without warmup)
    if args.start_warmup:
        scheduler = GradualWarmupScheduler(
            optimizer, 
            multiplier=1, 
            total_epoch=args.warmup_epochs, 
            after_scheduler=scheduler_step
        )
        # Set main scheduler last_epoch for resumed training
        if args.start_epoch > 0:
            scheduler.last_epoch = args.start_epoch - 1
    else:
        scheduler = scheduler_step

    return scheduler

def init_loss(args):
    L1_weight = args.L1_weight
    D_weight = args.D_weight 
    E_weight = args.E_weight 
    P_weight = 1.0
    
    # Use specific device if provided, otherwise use rank-based device
    device = dist.get_device()
    
    L1_loss= L1Loss(loss_weight=L1_weight, reduction='mean').to(device)
    D_loss = SSIM(weight=D_weight).to(device)
    E_loss = EdgeLoss(loss_weight=E_weight).to(device)
    P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = P_weight ,criterion='mse').to(device)
    return L1_loss,P_loss,E_loss,D_loss

def train(rank, args):
    if rank is not None:
        dist.init_distributed(rank)

    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    with Tee(os.path.join(f"./weights/train{now}", f'log.txt')):
        print(args)

        training_data_loader, testing_data_loader = load_datasets(args)
        model = build_model()
        L1_loss,P_loss,E_loss,D_loss = init_loss(args)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # Load checkpoint if start_epoch > 0
        start_epoch = 0
        if args.start_epoch > 0:
            pth = f"./weights/train/epoch_{args.start_epoch}.pth"
            checkpoint_data = torch.load(pth, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            print(f"Loaded checkpoint with optimizer state from epoch {checkpoint_data['epoch']}")
            start_epoch = args.start_epoch

        
        # Create scheduler after loading optimizer state
        scheduler = make_scheduler(optimizer, args)
        

        # Wrap for distributed training if available
        if dist.is_dist_available_and_initialized():
            training_data_loader = dist.warp_loader(training_data_loader, args.shuffle)
            model = dist.warp_model(model, find_unused_parameters=True, sync_bn=True)


        # train
        psnr = []
        ssim = []
        lpips = []

        for epoch in range(start_epoch+1, args.nEpochs + start_epoch + 1):
            # Set epoch for distributed sampler
            if dist.is_dist_available_and_initialized():
                training_data_loader.sampler.set_epoch(epoch)
                
            epoch_loss, batch_count = train_one_epoch(model, optimizer, training_data_loader, args, L1_loss, P_loss, E_loss, D_loss)
            scheduler.step()
            # Log basic epoch info for all processes
            avg_loss = epoch_loss / batch_count
            print("===> Epoch[{}] Avg Loss: {:.6f} || Learning rate: {:.6f}".format(
                epoch, avg_loss, optimizer.param_groups[0]['lr']))
            
            if epoch % args.snapshots == 0 and dist.is_main_process():
                checkpoint(epoch, model, optimizer, f"./weights/train{now}")
            
                avg_psnr, avg_ssim, avg_lpips = eval(model, testing_data_loader, args, alpha_i=1, use_GT_mean=args.use_GT_mean)
                print("===> Evaluation - PSNR: {:.4f} dB || SSIM: {:.4f} || LPIPS: {:.4f}".format(avg_psnr, avg_ssim, avg_lpips))
                psnr.append(avg_psnr)
                ssim.append(avg_ssim)
                lpips.append(avg_lpips)
                print(psnr)
                print(ssim)
                print(lpips)

            torch.cuda.empty_cache()

if __name__ == '__main__':
    args = option().parse_args()

    if args.gpu_mode == False:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""

    world_size = torch.cuda.device_count()
    print(f"Detected {world_size} GPUs")

    if world_size > 1:
        import torch.multiprocessing as mp
        mp.spawn(train, args=(args,), nprocs=world_size, join=True)
    else:
        train(None, args)