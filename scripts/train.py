import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import random
import inspect
from torchvision import transforms
import torch.optim as optim
import importlib
from scripts.options import option, load_datasets
from scripts.eval import eval
from data.data import *
from loss.losses import CIDNetCombinedLoss, CIDNetWithIntermediateLoss
from data.scheduler import *
from datetime import datetime
from scripts.measure import metrics
import dist
from scripts.utils import Tee, checkpoint, compute_model_complexity
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

    
def train_one_epoch(model, optimizer, training_data_loader, args, loss_f):
    import time
    start_time = time.time()
    model.train()
    total_loss = 0
    total_batches = 0
    torch.autograd.set_detect_anomaly(args.grad_detect)
    device = dist.get_device()
    
    for batch_idx, batch in enumerate(training_data_loader, 1):
        im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3]
        im1 = im1.to(device)
        im2 = im2.to(device)
        
        # use random gamma function (enhancement curve) to improve generalization
        if args.use_random_gamma:
            gamma = random.randint(args.start_gamma, args.end_gamma) / 100.0
            output_rgb, output_rgb_base = model(im1 ** gamma)  
        else:
            output_rgb, output_rgb_base = model(im1)
        
        # 통합 loss 계산 (RGB/HVI + Intermediate Supervision 모두 포함)
        loss = loss_f(output_rgb, output_rgb_base, im2)
        
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_batches += 1
        
    # 에폭 완료 후 샘플 이미지 저장 (마지막 배치의 결과)
    output_img = transforms.ToPILImage()((output_rgb)[0].squeeze(0))
    gt_img = transforms.ToPILImage()((im2)[0].squeeze(0))
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

def init_loss(args, trans):
    """
    Loss 함수 초기화
    
    Args:
        args: 학습 설정
        trans: RGB_to_HVI, HVI_to_RGB 변환 객체
    
    Returns:
        loss_fn
    """
    device = dist.get_device()
    
    # 기본 loss 생성
    base_loss = CIDNetCombinedLoss(
        trans=trans,
        L1_weight=args.L1_weight,
        D_weight=args.D_weight,
        E_weight=args.E_weight,
        P_weight=args.P_weight,
        HVI_weight=args.HVI_weight,
        use_gt_mean_loss=args.use_gt_mean_loss
    ).to(device)
    
    loss_fn = CIDNetWithIntermediateLoss(
        base_loss_fn=base_loss,
        intermediate_weight=args.intermediate_weight
    ).to(device)
    
    return loss_fn


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
        
        # Build model (class name matches file name)
        print('===> Building model ')
        model_file_path = args.model_file
        class_name = os.path.splitext(os.path.basename(model_file_path))[0]
        module_name = 'net.' + class_name
        module = importlib.import_module(module_name)
        ModelClass = getattr(module, class_name)
        params = inspect.signature(ModelClass.__init__).parameters
        if 'gamma' in params:
            # CIDNet_SSM uses 'gamma' for SSM scale range
            model = ModelClass(gamma=args.ssm_scale_range)
        else:
            model = ModelClass()
        model = model.to(dist.get_device())

        # Compute model complexity (only on main process, and BEFORE DDP wrapping)
        if dist.is_main_process():
            flops, params = compute_model_complexity(model, input_size=(1, 3, args.cropSize, args.cropSize))
            print(f"Model FLOPs: {flops}, Params: {params}")

        # Get trans object for loss calculation
        loss_fn = init_loss(args, model.trans)
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


        for epoch in range(start_epoch+1, args.nEpochs + start_epoch + 1):
            # Set epoch for distributed sampler
            if dist.is_dist_available_and_initialized():
                training_data_loader.sampler.set_epoch(epoch)
                
            avg_loss, epoch_time = train_one_epoch(model, optimizer, training_data_loader, args, loss_fn)
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