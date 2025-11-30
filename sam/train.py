import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import random
from torchvision import transforms
import torch.optim as optim
import importlib.util
from sam.options import option, load_datasets
from sam.eval import eval
from data.data import *
from loss.losses import *
from data.scheduler import *
from datetime import datetime
from measure import metrics
import dist
from sam.utils import Tee, checkpoint, compute_model_complexity
from torch.utils.tensorboard import SummaryWriter


def init_seed(seed, deterministic=False, benchmark=True):
    print(f"Using seed: {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)    
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = benchmark
    return seed


def train_one_epoch(model, optimizer, training_data_loader, args, L1_loss, P_loss, E_loss, D_loss):
    import time
    start_time = time.time()
    model.train()
    total_loss = 0  # 전체 에폭의 손실 합계
    total_batches = 0  # 전체 배치 수
    torch.autograd.set_detect_anomaly(args.grad_detect)
    device = dist.get_device()
    
    # Get RGB_to_HVI method from the actual model (handle DDP wrapper)
    actual_model = model.module if hasattr(model, 'module') else model
    rgb_to_hvi_fn = actual_model.RGB_to_HVI
    
    for batch_idx, batch in enumerate(training_data_loader, 1):
        im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3]

        # Use rank-based device assignment for distributed training
        
    
        im1 = im1.to(device)
        im2 = im2.to(device)
        
        # use random gamma function (enhancement curve) to improve generalization
        if args.gamma:
            gamma = random.randint(args.start_gamma,args.end_gamma) / 100.0
            output_rgb = model(im1 ** gamma)  
        else:
            output_rgb = model(im1)  
            
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
        
    # 에폭 완료 후 샘플 이미지 저장 (마지막 배치의 결과)
    output_img = transforms.ToPILImage()((output_rgb)[0].squeeze(0))
    gt_img = transforms.ToPILImage()((gt_rgb)[0].squeeze(0))
    if not os.path.exists(args.val_folder+'training'):          
        os.mkdir(args.val_folder+'training') 
    output_img.save(args.val_folder+'training/test.png')
    gt_img.save(args.val_folder+'training/gt.png')
    
    avg_loss = total_loss / total_batches
    elapsed_time = time.time() - start_time
    return avg_loss, elapsed_time


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

    with Tee(os.path.join(save_dir, f'1log.txt')):
        init_seed(args.seed + (rank if rank is not None else 0))

        # Initialize TensorBoard writer (only on main process)
        writer = None
        if dist.is_main_process():
            log_dir = os.path.join(save_dir, 'tensorboard')
            writer = SummaryWriter(log_dir)
            print(f"TensorBoard logging to: {log_dir}")

        print(args)

        training_data_loader, testing_data_loader = load_datasets(args)
        
        # Build model (always use CIDNet class from file)
        print('===> Building model ')
        
        model_file_path = args.model_file
        module_name = os.path.splitext(os.path.basename(model_file_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, model_file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ModelClass = getattr(module, 'CIDNet')
        # If CIDNet supports max_scale_factor, pass it
        import inspect
        params = inspect.signature(ModelClass.__init__).parameters
        if 'max_scale_factor' in params:
            model = ModelClass(max_scale_factor=args.max_scale_factor)
        else:
            model = ModelClass()
        model = model.to(dist.get_device())

        # Compute model complexity (only on main process, and BEFORE DDP wrapping)
        if dist.is_main_process():
            flops, params = compute_model_complexity(model)
            print(f"Model FLOPs: {flops}, Params: {params}")

        L1_loss, P_loss, E_loss, D_loss = init_loss(args)
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
            model = dist.warp_model(model, sync_bn=True, find_unused_parameters=True)


        # train
        psnr = []
        ssim = []
        lpips = []
        
        def eval_and_log(use_GT_mean=False):
            output_list, gt_list = eval(model, testing_data_loader, alpha_predict=False, base_alpha_s=1.3, base_alpha_i=1.0)
            avg_psnr, avg_ssim, avg_lpips = metrics(output_list, gt_list, use_GT_mean=use_GT_mean)
            print("===> Evaluation (use_GT_mean={}, alpha_predict=False, base_alpha_s=1.3, base_alpha_i=1.0) - PSNR: {:.4f} dB || SSIM: {:.4f} || LPIPS: {:.4f}".format(use_GT_mean, avg_psnr, avg_ssim, avg_lpips))
            
            output_list, gt_list = eval(model, testing_data_loader, alpha_predict=False, base_alpha_s=1.0, base_alpha_i=1.0)
            avg_psnr, avg_ssim, avg_lpips = metrics(output_list, gt_list, use_GT_mean=use_GT_mean)
            print("===> Evaluation (use_GT_mean={}, alpha_predict=False, base_alpha_s=1.0, base_alpha_i=1.0) - PSNR: {:.4f} dB || SSIM: {:.4f} || LPIPS: {:.4f}".format(use_GT_mean, avg_psnr, avg_ssim, avg_lpips))
            
            output_list, gt_list = eval(model, testing_data_loader, alpha_predict=True, base_alpha_s=1.3, base_alpha_i=1.0)
            avg_psnr, avg_ssim, avg_lpips = metrics(output_list, gt_list, use_GT_mean=use_GT_mean)
            print("===> Evaluation (use_GT_mean={}, alpha_predict=True, base_alpha_s=1.3, base_alpha_i=1.0) - PSNR: {:.4f} dB || SSIM: {:.4f} || LPIPS: {:.4f}".format(use_GT_mean, avg_psnr, avg_ssim, avg_lpips))
            
            output_list, gt_list = eval(model, testing_data_loader, alpha_predict=True, base_alpha_s=1.0, base_alpha_i=1.0)
            avg_psnr, avg_ssim, avg_lpips = metrics(output_list, gt_list, use_GT_mean=use_GT_mean)
            print("===> Evaluation (use_GT_mean={}, alpha_predict=True, base_alpha_s=1.0, base_alpha_i=1.0) - PSNR: {:.4f} dB || SSIM: {:.4f} || LPIPS: {:.4f}".format(use_GT_mean, avg_psnr, avg_ssim, avg_lpips))
            
            return avg_psnr, avg_ssim, avg_lpips
        
        # eval_and_log(False)
        # eval_and_log(True)

        for epoch in range(start_epoch+1, args.nEpochs + start_epoch + 1):
            # Set epoch for distributed sampler
            if dist.is_dist_available_and_initialized():
                training_data_loader.sampler.set_epoch(epoch)
                
            avg_loss, epoch_time = train_one_epoch(model, optimizer, training_data_loader, args, L1_loss, P_loss, E_loss, D_loss)
            scheduler.step()
            
            # Log basic epoch info for all processes
            print("===> Epoch[{}] Avg Loss: {:.6f} || Learning rate: {:.6f} || Time: {:.2f}s".format(
                epoch, avg_loss, optimizer.param_groups[0]['lr'], epoch_time))
            
            # Log to TensorBoard
            if writer is not None:
                writer.add_scalar('Loss/train', avg_loss, epoch)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            if epoch % args.snapshots == 0 and dist.is_main_process():
                checkpoint(epoch, model, optimizer, save_dir)

                eval_and_log(False)
                avg_psnr, avg_ssim, avg_lpips = eval_and_log(True)
                
                
                # Log evaluation metrics to TensorBoard
                if writer is not None:
                    writer.add_scalar('Metrics/PSNR', avg_psnr, epoch)
                    writer.add_scalar('Metrics/SSIM', avg_ssim, epoch)
                    writer.add_scalar('Metrics/LPIPS', avg_lpips, epoch)

            torch.cuda.empty_cache()
        
        # Close TensorBoard writer
        if writer is not None:
            writer.close()

if __name__ == '__main__':
    args = option().parse_args()

    if args.gpu_mode == False:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

    world_size = torch.cuda.device_count()
    print(f"Detected {world_size} GPUs")

    if world_size > 1:
        import torch.multiprocessing as mp
        mp.spawn(train, args=(args,), nprocs=world_size, join=True)
    else:
        train(None, args)