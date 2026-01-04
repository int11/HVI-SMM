import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import random
from torchvision import transforms
import torch.optim as optim
from net.CIDNet_SSM import CIDNet as CIDNet_SSM
from data.options import option, load_datasets
from sam.eval import eval
from data.data import *
from loss.losses import *
from data.scheduler import *
from datetime import datetime
from sam.measure import metrics
import dist
from sam.utils import Tee, checkpoint


def train_one_epoch(model, optimizer, cidnet_loader, ssm_loader, args, L1_loss, P_loss, E_loss, D_loss):
    model.train()
    total_loss = 0  # 전체 에폭의 손실 합계
    total_batches = 0  # 전체 배치 수
    torch.autograd.set_detect_anomaly(args.grad_detect)
    device = dist.get_device()
    
    # Get RGB_to_HVI method from the actual model (handle DDP wrapper)
    actual_model = model.module if hasattr(model, 'module') else model
    rgb_to_hvi_fn = actual_model.RGB_to_HVI
    
    # Phase 1: Train CIDNet body (SSM frozen, alpha_predict=False)
    # Freeze SSM
    for param in actual_model.alpha_predictor.parameters():
        param.requires_grad = False
    # Unfreeze CIDNet body
    for name, param in actual_model.named_parameters():
        if 'alpha_predictor' not in name:
            param.requires_grad = True
    
    for batch_idx, batch in enumerate(cidnet_loader, 1):
        im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3]
        im1 = im1.to(device)
        im2 = im2.to(device)
        
        # use random gamma function (enhancement curve) to improve generalization
        if args.gamma:
            gamma = random.randint(args.start_gamma, args.end_gamma) / 100.0
            output_rgb = model(im1 ** gamma, alpha_predict=False)
        else:
            output_rgb = model(im1, alpha_predict=False)
        
        gt_rgb = im2
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
        
        total_loss += loss.item()
        total_batches += 1
    
    # Phase 2: Train SSM only (CIDNet frozen, alpha_predict=True)
    # Freeze CIDNet body
    for name, param in actual_model.named_parameters():
        if 'alpha_predictor' not in name:
            param.requires_grad = False
    # Unfreeze SSM
    for param in actual_model.alpha_predictor.parameters():
        param.requires_grad = True
    
    for batch_idx, batch in enumerate(ssm_loader, 1):
        im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3]
        im1 = im1.to(device)
        im2 = im2.to(device)
        
        if args.gamma:
            gamma = random.randint(args.start_gamma, args.end_gamma) / 100.0
            output_rgb = model(im1 ** gamma, alpha_predict=True)
        else:
            output_rgb = model(im1, alpha_predict=True)
        
        gt_rgb = im2
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
    
    # Save sample image
    output_img = transforms.ToPILImage()((output_rgb)[0].squeeze(0).detach().cpu())
    gt_img = transforms.ToPILImage()((gt_rgb)[0].squeeze(0).detach().cpu())
    if not os.path.exists(args.val_folder+'training'):          
        os.mkdir(args.val_folder+'training') 
    output_img.save(args.val_folder+'training/test.png')
    gt_img.save(args.val_folder+'training/gt.png')
    
    return total_loss, total_batches
                


def build_model(max_scale_factor=1.2):
    print('===> Building model ')
    model = CIDNet_SSM(max_scale_factor=max_scale_factor)
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

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./weights/{args.dataset}/{now}"
    with Tee(os.path.join(save_dir, f'log.txt')):
        print(args)

        training_data_loader, testing_data_loader = load_datasets(args)
        
        # Split dataset: 60% for CIDNet, 40% for SSM
        from torch.utils.data import Subset, DataLoader
        dataset = training_data_loader.dataset
        total_samples = len(dataset)
        cidnet_split = int(total_samples * 0.6)
        
        all_indices = list(range(total_samples))
        random.shuffle(all_indices)
        
        cidnet_indices = all_indices[:cidnet_split]
        ssm_indices = all_indices[cidnet_split:]
        
        print(f"Total samples: {total_samples}")
        print(f"CIDNet training (60%, alpha_predict=False): {len(cidnet_indices)} samples")
        print(f"SSM training (40%, alpha_predict=True): {len(ssm_indices)} samples")
        
        # Create separate DataLoaders
        cidnet_dataset = Subset(dataset, cidnet_indices)
        ssm_dataset = Subset(dataset, ssm_indices)
        
        cidnet_loader = DataLoader(
            cidnet_dataset,
            batch_size=args.batchSize,
            shuffle=False,
            num_workers=args.threads,
            pin_memory=True
        )
        ssm_loader = DataLoader(
            ssm_dataset,
            batch_size=args.batchSize,
            shuffle=False,
            num_workers=args.threads,
            pin_memory=True
        )
        
        model = build_model(args.max_scale_factor)
        L1_loss,P_loss,E_loss,D_loss = init_loss(args)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # Load checkpoint if start_epoch > 0
        start_epoch = 0
        if args.start_epoch > 0:
            pth = f"./weights/train/epoch_{args.start_epoch}.pth"
            checkpoint_data = torch.load(pth, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            cidnet_indices = checkpoint_data.get('cidnet_indices', cidnet_indices)
            ssm_indices = checkpoint_data.get('ssm_indices', ssm_indices)
            if 'cidnet_indices' in checkpoint_data:
                print("Restored split indices from checkpoint")
            print(f"Loaded checkpoint from epoch {checkpoint_data['epoch']}")
            start_epoch = args.start_epoch

        
        # Create scheduler after loading optimizer state
        scheduler = make_scheduler(optimizer, args)
        

        # Wrap for distributed training if available
        if dist.is_dist_available_and_initialized():
            cidnet_loader = dist.warp_loader(cidnet_loader, shuffle=True)
            ssm_loader = dist.warp_loader(ssm_loader, shuffle=True)
            model = dist.warp_model(model, find_unused_parameters=True, sync_bn=True)


        # train
        psnr = []
        ssim = []
        lpips = []
        
        def eval_and_log(use_GT_mean=False):
            output_list, gt_list = eval(model, testing_data_loader, alpha_predict=False, base_alpha_s=1.3, base_alpha_i=1.0, alpha_rgb=1.0)
            avg_psnr, avg_ssim, avg_lpips = metrics(output_list, gt_list, use_GT_mean=use_GT_mean)
            print("===> Evaluation (use_GT_mean={}, alpha_predict=False, base_alpha_s=1.3, base_alpha_i=1.0) - PSNR: {:.4f} dB || SSIM: {:.4f} || LPIPS: {:.4f}".format(use_GT_mean, avg_psnr, avg_ssim, avg_lpips))
            
            output_list, gt_list = eval(model, testing_data_loader, alpha_predict=False, base_alpha_s=1.0, base_alpha_i=1.0, alpha_rgb=1.0)
            avg_psnr, avg_ssim, avg_lpips = metrics(output_list, gt_list, use_GT_mean=use_GT_mean)
            print("===> Evaluation (use_GT_mean={}, alpha_predict=False, base_alpha_s=1.0, base_alpha_i=1.0) - PSNR: {:.4f} dB || SSIM: {:.4f} || LPIPS: {:.4f}".format(use_GT_mean, avg_psnr, avg_ssim, avg_lpips))
            
            output_list, gt_list = eval(model, testing_data_loader, alpha_predict=True, base_alpha_s=1.3, base_alpha_i=1.0, alpha_rgb=1.0)
            avg_psnr, avg_ssim, avg_lpips = metrics(output_list, gt_list, use_GT_mean=use_GT_mean)
            print("===> Evaluation (use_GT_mean={}, alpha_predict=True, base_alpha_s=1.3, base_alpha_i=1.0) - PSNR: {:.4f} dB || SSIM: {:.4f} || LPIPS: {:.4f}".format(use_GT_mean, avg_psnr, avg_ssim, avg_lpips))
            
            output_list, gt_list = eval(model, testing_data_loader, alpha_predict=True, base_alpha_s=1.0, base_alpha_i=1.0, alpha_rgb=1.0)
            avg_psnr, avg_ssim, avg_lpips = metrics(output_list, gt_list, use_GT_mean=use_GT_mean)
            print("===> Evaluation (use_GT_mean={}, alpha_predict=True, base_alpha_s=1.0, base_alpha_i=1.0) - PSNR: {:.4f} dB || SSIM: {:.4f} || LPIPS: {:.4f}".format(use_GT_mean, avg_psnr, avg_ssim, avg_lpips))
            
            return avg_psnr, avg_ssim, avg_lpips
        
        # eval_and_log(False)
        # eval_and_log(True)

        for epoch in range(start_epoch+1, args.nEpochs + start_epoch + 1):
            # Set epoch for distributed sampler
            if dist.is_dist_available_and_initialized():
                cidnet_loader.sampler.set_epoch(epoch)
                ssm_loader.sampler.set_epoch(epoch)
                
            epoch_loss, batch_count = train_one_epoch(model, optimizer, cidnet_loader, ssm_loader, args, L1_loss, P_loss, E_loss, D_loss)
            scheduler.step()
            # Log basic epoch info for all processes
            avg_loss = epoch_loss / batch_count
            print("===> Epoch[{}] Avg Loss: {:.6f} || Learning rate: {:.6f}".format(
                epoch, avg_loss, optimizer.param_groups[0]['lr']))
            
            if epoch % args.snapshots == 0 and dist.is_main_process():
                actual_model = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': actual_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'cidnet_indices': cidnet_indices,
                    'ssm_indices': ssm_indices,
                }, os.path.join(save_dir, f'epoch_{epoch}.pth'))

                eval_and_log(False)
                avg_psnr, avg_ssim, avg_lpips = eval_and_log(True)
                
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