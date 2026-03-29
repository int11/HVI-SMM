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
from loss.hvi_losses import CIDNetCombinedLoss, CIDNetWithIntermediateLoss
from data.scheduler import *
from datetime import datetime
from scripts.measure import metrics
import scripts.dist as dist
from scripts.utils import Tee, checkpoint, compute_model_complexity, init_seed, build_generator_from_args, make_common_scheduler
from torch.utils.tensorboard import SummaryWriter
from net.BaseCIDNetWithSMM import BaseCIDNet_SMM


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
            output = model(im1 ** gamma)  
        else:
            output = model(im1)
        
        if not isinstance(output, tuple):
            output = (output,)

        # if cidnet_smm,  CIDNetWithIntermediateLoss(output_rgb, output_rgb_base, im2) elif cidnet CIDNetCombinedLoss(output_rgb, im2)
        loss = loss_f(*output, im2)

        
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_batches += 1
        
    # 에폭 완료 후 샘플 이미지 저장 (마지막 배치의 결과)
    output_img = transforms.ToPILImage()((output[0])[0].squeeze(0))
    gt_img = transforms.ToPILImage()((im2)[0].squeeze(0))
    if not os.path.exists(args.val_folder+'training'):          
        os.makedirs(args.val_folder+'training') 
    output_img.save(args.val_folder+'training/test.png')
    gt_img.save(args.val_folder+'training/gt.png')
    
    avg_loss = total_loss / total_batches
    elapsed_time = time.time() - start_time
    return avg_loss, elapsed_time


def init_loss(args, trans, model):
    """
    Loss 함수 초기화
    
    Args:
        args: 학습 설정
        trans: RGB_to_HVI, HVI_to_RGB 변환 객체
        model: 모델 인스턴스
    
    Returns:
        loss_fn
    """
    device = dist.get_device()
    
    # 기본 loss 생성
    base_loss = CIDNetCombinedLoss(
        trans=trans,
        l1_weight=args.l1_weight,
        d_weight=args.d_weight,
        e_weight=args.e_weight,
        p_weight=args.p_weight,
        hvi_weight=args.hvi_weight,
        use_gt_mean_loss=args.use_gt_mean_loss
    ).to(device)
    
    if isinstance(model, BaseCIDNet_SMM):
        # CIDNet_SSM인 경우 Intermediate Supervision loss 포함
        loss_fn = CIDNetWithIntermediateLoss(
            base_loss_fn=base_loss,
            intermediate_weight=args.intermediate_weight
        ).to(device)
    else:
        loss_fn = base_loss
    
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
        
        # Build model
        print('===> Building model ')
        model = build_generator_from_args(args).to(dist.get_device())

        # Compute model complexity (only on main process, and BEFORE DDP wrapping)
        if dist.is_main_process():
            flops, params = compute_model_complexity(model, input_size=(1, 3, args.crop_size, args.crop_size))
            print(f"Model FLOPs: {flops}, Params: {params}")

        # Get trans object for loss calculation
        loss_fn = init_loss(args, model.trans, model)
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
        scheduler = make_common_scheduler(optimizer, args)
        

        # Wrap for distributed training if available
        if dist.is_dist_available_and_initialized():
            training_data_loader = dist.warp_loader(training_data_loader, args.shuffle)
            model = dist.warp_model(model, sync_bn=True, find_unused_parameters=True)


        # train
        psnr = []
        ssim = []
        lpips = []
        

        
        def eval_and_log_batched(alpha_combinations):
            """
            Evaluate all alpha combinations in a single batched forward pass.
            
            Args:
                alpha_combinations: List of (base_alpha_s, base_alpha_i, alpha_rgb) tuples
            
            Returns:
                metrics for the last alpha combination (for backward compatibility)
            """
            # Batched evaluation - computes all alpha combinations efficiently
            results = eval(model, testing_data_loader, alpha_combinations)
            
            # Process each alpha combination
            last_metrics = None
            for (base_alpha_s, base_alpha_i, alpha_rgb) in alpha_combinations:
                output_list, gt_list = results[(base_alpha_s, base_alpha_i, alpha_rgb)]
                
                # (1) Compute metrics with use_gt_mean=True
                avg_psnr_gt, avg_ssim_gt, avg_lpips_gt = metrics(output_list, gt_list, use_gt_mean=True)
                print("===> Evaluation (use_gt_mean=True, base_alpha_s={}, base_alpha_i={}, alpha_rgb={}) - PSNR: {:.4f} dB || SSIM: {:.4f} || LPIPS: {:.4f}".format(
                    base_alpha_s, base_alpha_i, alpha_rgb, avg_psnr_gt, avg_ssim_gt, avg_lpips_gt))
                
                # (2) Compute metrics with use_gt_mean=False
                avg_psnr, avg_ssim, avg_lpips = metrics(output_list, gt_list, use_gt_mean=False)
                print("===> Evaluation (use_gt_mean=False, base_alpha_s={}, base_alpha_i={}, alpha_rgb={}) - PSNR: {:.4f} dB || SSIM: {:.4f} || LPIPS: {:.4f}".format(
                    base_alpha_s, base_alpha_i, alpha_rgb, avg_psnr, avg_ssim, avg_lpips))
                
                last_metrics = (avg_psnr, avg_ssim, avg_lpips)
            
            return last_metrics


        for epoch in range(start_epoch+1, args.n_epochs + start_epoch + 1):
            # Set epoch for distributed sampler
            if dist.is_dist_available_and_initialized():
                training_data_loader.sampler.set_epoch(epoch)
                
            avg_loss, epoch_time = train_one_epoch(model, optimizer, training_data_loader, args, loss_fn)
            if scheduler is not None:
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

                # Evaluate all alpha combinations in a single batched pass
                alpha_combinations = [(1.3, 1.0, 1.0), (1.0, 1.0, 0.8), (1.0, 1.0, 1.0)]
                avg_psnr, avg_ssim, avg_lpips = eval_and_log_batched(alpha_combinations)
                
                
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