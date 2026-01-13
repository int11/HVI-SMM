import torch
import numpy as np
import random
import os
from torch.utils.data import DataLoader
import torch.nn as nn 


def init_distributed(rank, print_rank: int=0, print_method: str='builtin'):
    """
    env setup
    args:
        print_rank,
        print_method, (builtin, rich)
        seed,
    """
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"  # pick a free port
        torch.distributed.init_process_group("nccl", rank=rank, world_size=torch.cuda.device_count())
        torch.cuda.set_device(rank)
        torch.distributed.barrier()


        torch.cuda.empty_cache()
        enabled_dist = True
        print(f'Initialized distributed mode...Rank {rank}')

    except Exception as e:
        print(e)
        enabled_dist = False
        print('Not init distributed mode.')

    setup_print(get_device() == print_rank, method=print_method)

    return enabled_dist

def get_device():
    """일관된 타입으로 디바이스 반환 (정수 또는 'cpu' 문자열)"""
    return torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

def is_main_process():
    return get_device() == 0 or get_device() == 'cpu'

def setup_print(is_main, method='builtin'):
    """This function disables printing when not in master process
    """
    import builtins as __builtin__

    if method == 'builtin':
        builtin_print = __builtin__.print

    elif method == 'rich':
        import rich 
        builtin_print = rich.print

    else:
        raise AttributeError('')

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def warp_loader(loader, shuffle):       
    sampler = torch.utils.data.DistributedSampler(loader.dataset, shuffle=shuffle)
    loader = DataLoader(loader.dataset,
                        loader.batch_size,
                        sampler=sampler,
                        shuffle=False,
                        drop_last=loader.drop_last,
                        collate_fn=loader.collate_fn,
                        pin_memory=loader.pin_memory,
                        num_workers=loader.num_workers)
    return loader


def warp_model(model, sync_bn: bool=False, find_unused_parameters: bool=False, **kwargs):

    rank = get_device()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) if sync_bn else model 
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=find_unused_parameters)

    return model


def is_dist_available_and_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def is_parallel(model) -> bool:
    # Returns True if model is of type DP or DDP
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)


def de_parallel(model) -> nn.Module:
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model