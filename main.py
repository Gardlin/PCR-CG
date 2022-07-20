import os, torch, time, shutil, json,glob, argparse, shutil
import numpy as np
from easydict import EasyDict as edict

from datasets.dataloader import get_dataloader, get_datasets
from models.architectures import KPFCNN
from lib.utils import setup_seed, load_config
from lib.tester import get_trainer
from lib.loss import MetricLoss
from configs.models import architectures

from torch import optim
from torch import nn
setup_seed(0)
os.environ['CUDA_VISIBLE_DEVICES']="0"

if __name__ == '__main__':
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train/indoor.yaml', help= 'Path to the config file.')
    args = parser.parse_args()
    config = load_config(args.config)
    config = edict(config)

    snapshot=os.path.join(f'{config.dir}',f'{config.benchmark}',f'{config.num_gpus}_gpu_{config.img_num}_img_initmode_{config.init_mode}_{config.first_feats_dim}_gamma{config.scheduler_gamma}_lr{config.lr}_finalfeatsdim_{config.final_feats_dim}',f'{config.mode}')
    print(f"save results to {snapshot}")
    config['snapshot_dir'] = f'{snapshot}'
    config['tboard_dir'] = f'{snapshot}/tensorboard'
    config['save_dir'] = f'{snapshot}/checkpoints'
    config = edict(config)

    os.makedirs(config.snapshot_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.tboard_dir, exist_ok=True)
    json.dump(
        config,
        open(os.path.join(config.snapshot_dir, 'config.json'), 'w'),
        indent=4,
    )
    if torch.cuda.is_available():
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')
    
    # backup the files
    os.system(f'cp -r models {config.snapshot_dir}')
    os.system(f'cp -r datasets {config.snapshot_dir}')
    os.system(f'cp -r lib {config.snapshot_dir}')
    os.system(f'cp -r configs {config.snapshot_dir}')
    shutil.copy2('main.py',config.snapshot_dir)
    shutil.copy2('configs/train/indoor.yaml',config.snapshot_dir)

    # model initialization
    config.architecture = architectures[config.dataset]
    config.model = KPFCNN(config)


    # create optimizer 
    if config.optimizer == 'SGD':
        config.optimizer = optim.SGD(
            config.model.parameters(), 
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            )
    elif config.optimizer == 'ADAM':
        config.optimizer = optim.Adam(
            config.model.parameters(), 
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
    
    # create learning rate scheduler
    config.scheduler = optim.lr_scheduler.ExponentialLR(
        config.optimizer,
        gamma=config.scheduler_gamma,
    )
    
    # create dataset and dataloader
    train_set, val_set, benchmark_set = get_datasets(config)
    config.train_loader, neighborhood_limits = get_dataloader(dataset=train_set,
                                        batch_size=config.batch_size,
                                        shuffle=True,
                                        num_workers=config.num_workers,
                                        )
    config.val_loader, _ = get_dataloader(dataset=val_set,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=1,
                                        neighborhood_limits=neighborhood_limits
                                        )
    config.test_loader, _ = get_dataloader(dataset=benchmark_set,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=1,
                                        neighborhood_limits=neighborhood_limits)
    
    # create evaluation metrics
    config.desc_loss = MetricLoss(config)
    trainer = get_trainer(config)
    os.environ["OMP_NUM_THREADS"] = "16"
    if(config.mode=='train'):
        trainer.train()
    elif(config.mode =='val'):
        trainer.eval()
    else:
        trainer.test()        
