import os
import sys
import torch
import numpy as np
import random
import importlib
import inspect
import torchvision.transforms as transforms
from datetime import datetime
import scripts.dist as dist
from data.scheduler import CosineAnnealingRestartLR, GradualWarmupScheduler
from fvcore.nn import FlopCountAnalysis

def init_seed(seed, deterministic=False, benchmark=True):
    print(f"Using seed: {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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

def build_generator_from_args(args):
    """Dynamically load generator model from args.model_file."""
    model_file_path = args.model_file
    class_name = os.path.splitext(os.path.basename(model_file_path))[0]
    module_name = 'net.' + class_name
    module = importlib.import_module(module_name)
    ModelClass = getattr(module, class_name)
    
    params = inspect.signature(ModelClass.__init__).parameters
    if 'gamma' in params:
        model = ModelClass(gamma=args.ssm_scale_range)
    else:
        model = ModelClass()
    return model

def make_common_scheduler(optimizer, args):
    """Common scheduler logic shared between train.py and train_fofa.py."""
    cosine_last_epoch = -1 if args.start_epoch == 0 else args.start_epoch - 1 - args.warmup_epochs
    
    if args.scheduler == 'None':
        return None

    if args.scheduler == 'cos_restart':
        periods = [args.n_epochs - args.warmup_epochs] if args.start_warmup else [args.n_epochs]
        scheduler_step = CosineAnnealingRestartLR(
            optimizer=optimizer, 
            periods=periods, 
            restart_weights=[1], 
            eta_min=1e-7,
            last_epoch=cosine_last_epoch
        )
    else:
        # For train.py backward compatibility or other types
        # This can be extended as needed
        raise NotImplementedError(f"Scheduler {args.scheduler} not fully implemented in common factory yet.")
    
    if args.start_warmup:
        scheduler = GradualWarmupScheduler(
            optimizer, 
            multiplier=1, 
            total_epoch=args.warmup_epochs, 
            after_scheduler=scheduler_step
        )
        if args.start_epoch > 0:
            scheduler.last_epoch = args.start_epoch - 1
    else:
        scheduler = scheduler_step

    return scheduler

def compute_model_complexity(model, input_size=(1, 3, 384, 384)):
    """
    Compute model FLOPs and parameters using fvcore FlopCountAnalysis
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
    
    Returns:
        flops_str: Number of FLOPs (string format)
        params_str: Number of parameters (string format)
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Format parameters
    if total_params >= 1e9:
        params_str = f"{total_params / 1e9:.2f}G"
    elif total_params >= 1e6:
        params_str = f"{total_params / 1e6:.2f}M"
    elif total_params >= 1e3:
        params_str = f"{total_params / 1e3:.2f}K"
    else:
        params_str = f"{total_params}"
    
    # Get model device
    device = next(model.parameters()).device
    model.eval()
    
    # Use fvcore FlopCountAnalysis for accurate FLOPs calculation
    input_tensor = torch.randn(input_size, device=device)
    
    with torch.no_grad():
        flops_anal = FlopCountAnalysis(model, input_tensor)
        total_flops = flops_anal.total()
    
    # Format FLOPs
    if total_flops >= 1e12:
        flops_str = f"{total_flops / 1e12:.2f}T"
    elif total_flops >= 1e9:
        flops_str = f"{total_flops / 1e9:.2f}G"
    elif total_flops >= 1e6:
        flops_str = f"{total_flops / 1e6:.2f}M"
    elif total_flops >= 1e3:
        flops_str = f"{total_flops / 1e3:.2f}K"
    else:
        flops_str = f"{total_flops}"

    return flops_str, params_str

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


def checkpoint(epoch, model, optimizer, path, filename=None):
    os.makedirs(path, exist_ok=True)
    if filename is None:
        model_out_path = os.path.join(path, f"epoch_{epoch}.pth")
    else:
        model_out_path = os.path.join(path, filename)

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


def fofa_checkpoint(epoch, generator, discriminator,
                    optimizer_g, optimizer_d,
                    scheduler_g, scheduler_d, path):
    """
    FoFA \ube44\uc9c0\ub3c4 \ud559\uc2b5\uc6a9 \uccb4\ud06c\ud3ec\uc778\ud2b8 \uc800\uc7a5.\n    \uc0dd\uc131\uae30(G) + \ud310\ubcc4\uae30(D) + \uc591\ucabd optimizer/scheduler \uc0c1\ud0dc\ub97c \ud55c \ud30c\uc77c\uc5d0 \uc800\uc7a5\ud55c\ub2e4.
    """
    os.makedirs(path, exist_ok=True)
    out_path = os.path.join(path, f"epoch_{epoch}.pth")

    def _state(m):
        return (dist.de_parallel(m).state_dict()
                if dist.is_dist_available_and_initialized()
                else m.state_dict())

    ckpt = {
        'epoch': epoch,
        'generator_state_dict': _state(generator),
        'discriminator_state_dict': _state(discriminator),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'scheduler_g_state_dict': scheduler_g.state_dict() if scheduler_g is not None else None,
        'scheduler_d_state_dict': scheduler_d.state_dict() if scheduler_d is not None else None,
    }
    torch.save(ckpt, out_path)
    print(f"FoFA checkpoint saved to {out_path}")


def fofa_load_checkpoint(path, generator, discriminator,
                         optimizer_g, optimizer_d,
                         scheduler_g=None, scheduler_d=None,
                         device='cpu'):
    """
    fofa_checkpoint\ub85c \uc800\uc7a5\ub41c \uccb4\ud06c\ud3ec\uc778\ud2b8\ub97c \ubcf5\uc6d0\ud55c\ub2e4.

    Returns:
        start_epoch (int)
    """
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt['generator_state_dict'])
    discriminator.load_state_dict(ckpt['discriminator_state_dict'])
    optimizer_g.load_state_dict(ckpt['optimizer_g_state_dict'])
    optimizer_d.load_state_dict(ckpt['optimizer_d_state_dict'])
    if scheduler_g is not None and ckpt.get('scheduler_g_state_dict') is not None:
        scheduler_g.load_state_dict(ckpt['scheduler_g_state_dict'])
    if scheduler_d is not None and ckpt.get('scheduler_d_state_dict') is not None:
        scheduler_d.load_state_dict(ckpt['scheduler_d_state_dict'])
    start_epoch = ckpt['epoch']
    print(f"FoFA checkpoint loaded from {path} (epoch {start_epoch})")
    return start_epoch
