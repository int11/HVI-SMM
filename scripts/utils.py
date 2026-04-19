import os
import sys
import glob
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def plot_from_tfevents(save_dir: str, filename: str = 'training_plot.png'):
    """
    save_dir/tensorboard/ 의 events 파일을 읽어 PNG로 저장.
    매 epoch 끝에 writer.flush() 후 호출.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tb_dir = os.path.join(save_dir, 'tensorboard')
    event_files = glob.glob(os.path.join(tb_dir, 'events.out.tfevents.*'))
    if not event_files:
        return

    ea = EventAccumulator(event_files[0])
    ea.Reload()
    scalar_tags = ea.Tags().get('scalars', [])
    if not scalar_tags:
        return

    loss_tags   = [t for t in scalar_tags if t.startswith('Loss/')]
    metric_tags = [t for t in scalar_tags if t.startswith('Metrics/') or t.startswith('Eval/')]
    lr_tags     = [t for t in scalar_tags if 'LR' in t or 'Learning' in t]

    n_rows = 1 + (1 if metric_tags else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    # Loss plot
    ax = axes[0]
    for tag in loss_tags + lr_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        vals  = [e.value for e in events]
        style = '--' if tag in lr_tags else '-'
        ax.plot(steps, vals, style, label=tag.split('/')[-1], linewidth=1.2, alpha=0.85)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend(ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Metrics plot
    if metric_tags:
        ax  = axes[1]
        ax2 = ax.twinx()
        right_axis = {'SSIM', 'LPIPS'}
        for tag in metric_tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            vals  = [e.value for e in events]
            name  = tag.split('/')[-1]
            if name in right_axis:
                ax2.plot(steps, vals, 'o--', label=name, markersize=4, linewidth=1.2)
            else:
                ax.plot(steps, vals, 'o-', label=name, markersize=4, linewidth=1.2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('PSNR / NIQE / BRISQUE')
        ax2.set_ylabel('SSIM / LPIPS')
        ax.set_title('Eval Metrics')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, ncol=3, fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=120)
    plt.close(fig)


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
