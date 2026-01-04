import argparse
import random
import numpy as np
from torch.utils.data import DataLoader
from data.data import transform1, transform2, transform3
from data.LOLdataset import LOLv1DatasetFromFolder, LOLv2DatasetFromFolder, LOLv2SynDatasetFromFolder
from data.eval_sets import DatasetFromFolderEval, SICEDatasetFromFolderEval
from data.SICE_blur_SID import LOLBlurDatasetFromFolder, SIDDatasetFromFolder, SICEDatasetFromFolder


def str_to_bool(v):
    """Convert string to boolean"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')


def worker_init_fn(worker_id):
    """Initialize random seed for each worker"""
    worker_seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def option():
    # Training settings
    parser = argparse.ArgumentParser(description='CIDNet')
    parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
    parser.add_argument('--cropSize', type=int, default=384, help='image crop size (patch size)')
    parser.add_argument('--nEpochs', type=int, default=500, help='number of epochs to train for end')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to start, >0 is retrained a pre-trained pth')
    parser.add_argument('--snapshots', type=int, default=10, help='Snapshots for save checkpoints pth')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--gpu_mode', type=str_to_bool, default=True)
    parser.add_argument('--cuda_visible_devices', type=str, default="0", help='Set CUDA_VISIBLE_DEVICES (e.g., "0,1")')
    parser.add_argument('--shuffle', type=str_to_bool, default=True)
    parser.add_argument('--threads', type=int, default=4, help='number of threads for dataloader to use')
    parser.add_argument('--seed', type=int, default=1, help='random seed to use. Default=123')

    # choose a scheduler
    parser.add_argument('--model_file', type=str, default='net/CIDNet_SSM.py', help='Path to model file (e.g., net/CIDNet_fix.py)')
    parser.add_argument('--cos_restart_cyclic', type=str_to_bool, default=False)
    parser.add_argument('--cos_restart', type=str_to_bool, default=True)

    # warmup training
    parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup_epochs')
    parser.add_argument('--start_warmup', type=str_to_bool, default=True, help='turn False to train without warmup') 

    # choose which dataset you want to train, please only set one "True"
    parser.add_argument('--dataset', type=str, default='lolv2_syn', choices=['lol_v1', 'lolv2_real', 'lolv2_syn', 'lol_blur', 'SID', 'SICE_mix', 'SICE_grad'], help='Choose one dataset to train on')

    # train datasets
    parser.add_argument('--data_train_lol_blur'     , type=str, default='./datasets/LOL_blur/train')
    parser.add_argument('--data_train_lol_v1'       , type=str, default='./datasets/LOLdataset/our485')
    parser.add_argument('--data_train_lolv2_real'   , type=str, default='./datasets/LOL-v2/Real_captured/Train')
    parser.add_argument('--data_train_lolv2_syn'    , type=str, default='./datasets/LOL-v2/Synthetic/Train')
    parser.add_argument('--data_train_SID'          , type=str, default='./datasets/Sony_total_dark/train')
    parser.add_argument('--data_train_SICE'         , type=str, default='./datasets/SICE/Dataset/train')

    # validation input
    parser.add_argument('--data_val_lol_blur'       , type=str, default='./datasets/LOL_blur/eval/low_blur')
    parser.add_argument('--data_val_lol_v1'         , type=str, default='./datasets/LOLdataset/eval15')
    parser.add_argument('--data_val_lolv2_real'     , type=str, default='./datasets/LOL-v2/Real_captured/Test')
    parser.add_argument('--data_val_lolv2_syn'      , type=str, default='./datasets/LOL-v2/Synthetic/Test')
    parser.add_argument('--data_val_SID'            , type=str, default='./datasets/Sony_total_dark/eval/short')
    parser.add_argument('--data_val_SICE_mix'       , type=str, default='./datasets/SICE/Dataset/eval/test')
    parser.add_argument('--data_val_SICE_grad'      , type=str, default='./datasets/SICE/Dataset/eval/test')

    parser.add_argument('--val_folder', default='./results/', help='Location to save validation datasets')

    # loss weights
    parser.add_argument('--HVI_weight', type=float, default=1.0)
    parser.add_argument('--L1_weight', type=float, default=1.0)
    parser.add_argument('--D_weight',  type=float, default=0.5)
    parser.add_argument('--E_weight',  type=float, default=50.0)
    parser.add_argument('--P_weight',  type=float, default=1e-2)
    parser.add_argument('--intermediate_weight', type=float, default=0.5, help='Weight for intermediate supervision loss (I_base)')
    parser.add_argument('--use_gt_mean_loss', type=str, default='hvi', choices=['none', 'rgb', 'hvi'], help='Loss type: none (no GT-Mean Loss), rgb (GT-Mean Loss), hvi (Intensity Mean Loss)')
    
    # use random gamma function (enhancement curve) to improve generalization
    parser.add_argument('--use_random_gamma', type=str_to_bool, default=False, help='Use random gamma augmentation during training')
    parser.add_argument('--start_gamma', type=int, default=60, help='Start gamma value for augmentation (60 = 0.6)')
    parser.add_argument('--end_gamma', type=int, default=120, help='End gamma value for augmentation (120 = 1.2)')

    # SSM parameters
    parser.add_argument('--ssm_scale_range', type=float, default=0.5, help='SSM scale adjustment range (0.5 means [0.5x, 1.5x])')

    # auto grad, turn off to speed up training
    parser.add_argument('--grad_detect', type=str_to_bool, default=False, help='if gradient explosion occurs, turn-on it')
    parser.add_argument('--grad_clip', type=str_to_bool, default=True, help='if gradient fluctuates too much, turn-on it')
    return parser


def load_datasets(opt):
    print('===> Loading datasets')
    dataset = opt.dataset
    if dataset == 'lol_v1':
        train_set = LOLv1DatasetFromFolder(opt.data_train_lol_v1, transform=transform1(opt.cropSize))
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle, worker_init_fn=worker_init_fn)
        test_set = DatasetFromFolderEval(opt.data_val_lol_v1, folder1='low', folder2='high', transform=transform2())
        testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False, worker_init_fn=worker_init_fn)
    elif dataset == 'lol_blur':
        train_set = LOLBlurDatasetFromFolder(opt.data_train_lol_blur, transform=transform1(opt.cropSize))
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle, worker_init_fn=worker_init_fn)
        test_set = DatasetFromFolderEval(opt.data_val_lol_blur, folder1='low_blur', folder2='high_sharp_scaled', transform=transform2())
        testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False, worker_init_fn=worker_init_fn)
    elif dataset == 'lolv2_real':
        train_set = LOLv2DatasetFromFolder(opt.data_train_lolv2_real, transform=transform1(opt.cropSize))
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle, worker_init_fn=worker_init_fn)
        test_set = DatasetFromFolderEval(opt.data_val_lolv2_real, folder1='Low', folder2='Normal', transform=transform2())
        testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False, worker_init_fn=worker_init_fn)
    elif dataset == 'lolv2_syn':
        train_set = LOLv2SynDatasetFromFolder(opt.data_train_lolv2_syn, transform=transform3())
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle, worker_init_fn=worker_init_fn)
        test_set = DatasetFromFolderEval(opt.data_val_lolv2_syn, folder1='Low', folder2='Normal', transform=transform2())
        testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False, worker_init_fn=worker_init_fn)
    elif dataset == 'SID':
        train_set = SIDDatasetFromFolder(opt.data_train_SID, transform=transform1(opt.cropSize))
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle, worker_init_fn=worker_init_fn)
        test_set = DatasetFromFolderEval(opt.data_val_SID, folder1='short', folder2='long', transform=transform2())
        testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False, worker_init_fn=worker_init_fn)
    elif dataset == 'SICE_mix':
        train_set = SICEDatasetFromFolder(opt.data_train_SICE, transform=transform1(opt.cropSize))
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle, worker_init_fn=worker_init_fn)
        test_set = SICEDatasetFromFolderEval(opt.data_val_SICE_mix, transform=transform2())
        testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False, worker_init_fn=worker_init_fn)
    elif dataset == 'SICE_grad':
        train_set = SICEDatasetFromFolder(opt.data_train_SICE, transform=transform1(opt.cropSize))
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
        test_set = SICEDatasetFromFolderEval(opt.data_val_SICE_grad, transform=transform2())
        testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    else:
        raise Exception("should choose a valid dataset")
    return training_data_loader, testing_data_loader
