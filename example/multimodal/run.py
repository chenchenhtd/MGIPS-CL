import os
import hydra
import torch
import numpy as np
import random
from hydra import utils
from torch.utils.data import DataLoader
from deepke.name_entity_re.multimodal.models.IFA_model import MCIPSCLCRFModel
from deepke.name_entity_re.multimodal.modules.dataset import MMPNERProcessor, MMPNERDataset
from deepke.name_entity_re.multimodal.modules.train import Trainer

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

import wandb
# 创建Optuna的study对象，并指定需要优化的目标函数和搜索空间
import optuna

DATA_PATH = {
    'twitter15': {'train': 'data/twitter2015/train.txt',
                'dev': 'data/twitter2015/valid.txt',
                'test': 'data/twitter2015/test.txt',
                'train_auximgs': 'data/twitter2015/twitter2015_train_dict.pth',
                'dev_auximgs': 'data/twitter2015/twitter2015_val_dict.pth',
                'test_auximgs': 'data/twitter2015/twitter2015_test_dict.pth',
                'rcnn_img_path': 'data/twitter2015',
                'img2crop': 'data/twitter2015/twitter15_detect/twitter15_img2crop.pth'},

    'twitter17': {'train': 'data/twitter2017/train.txt',
                'dev': 'data/twitter2017/valid.txt',
                'test': 'data/twitter2017/test.txt',
                'train_auximgs': 'data/twitter2017/twitter2017_train_dict.pth',
                'dev_auximgs': 'data/twitter2017/twitter2017_val_dict.pth',
                'test_auximgs': 'data/twitter2017/twitter2017_test_dict.pth',
                'rcnn_img_path': 'data/twitter2017',
                'img2crop': 'data/twitter2017/twitter17_detect/twitter17_img2crop.pth'}
    }

IMG_PATH = {
    'twitter15': 'data/twitter2015/twitter2015_images',
    'twitter17': 'data/twitter2017/twitter2017_images'
}

AUX_PATH = {
    'twitter15': {'train': 'data/twitter2015/twitter2015_aux_images/train/crops',
                'dev': 'data/twitter2015/twitter2015_aux_images/val/crops',
                'test': 'data/twitter2015/twitter2015_aux_images/test/crops'},

    'twitter17': {'train': 'data/twitter2017/twitter2017_aux_images/train/crops',
                'dev': 'data/twitter2017/twitter2017_aux_images/val/crops',
                'test': 'data/twitter2017/twitter2017_aux_images/test/crops'}
}

# LABEL_LIST = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]", "X"]
LABEL_LIST = ["[CLS]","O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[SEP]", "X" ]
def set_seed(seed=2021):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

cfg = None

# 定义目标函数
def objective(trial):
    if cfg.use_wandb:
        writer = wandb.init(project="DeepKE_NER_MM")
    else:
        writer=None
    # cwd = utils.get_original_cwd()
    # cfg.cwd = cwd
    # weight_de = 7e-3
    # bert_lr,vit_lr = 5e-5,3e-5
    # hie_lr, con_lr = 3e-4,3e-3
    k = 3

    cfg.lr = trial.suggest_float('lr',5e-5, 6e-5) # 3e-5/5e-5
    cfg.vit_lr = trial.suggest_float('vit_lr',1e-5, 3e-5) # 3e-5/5e-5
    cfg.weight_decay = trial.suggest_float('weight_decay', 5e-3, 7e-3) # 3e-5/5e-5
    cfg.hie_lr = trial.suggest_float('hie_lr',3e-4, 6e-4) # 3e-5/5e-5
    cfg.con_lr = trial.suggest_float('con_lr',3e-3, 6e-3) # 3e-5/5e-5

    cfg.lr = 5.0829548841600325e-05
    cfg.vit_lr = 1.4978371199830381e-05
    cfg.weight_decay = 0.005729391146531671
    cfg.hie_lr = 0.0003492122598446973
    cfg.con_lr = 0.004484368729846639

    cfg.k = k
    print(cfg)

    set_seed(cfg.seed)  # set seed, default is 1
    if cfg.save_path is not None:  # make save_path dir
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path, exist_ok=True)
    print(cfg)
    label_mapping = {label: idx for idx, label in enumerate(LABEL_LIST, 0)}
    # label_mapping["PAD"] = 0
    data_path, img_path, aux_path = DATA_PATH[cfg.dataset_name], IMG_PATH[cfg.dataset_name], AUX_PATH[cfg.dataset_name]
    rcnn_img_path = DATA_PATH[cfg.dataset_name]['rcnn_img_path']
    rcnn_img_path = None

    processor = MMPNERProcessor(data_path, cfg)
    train_dataset = MMPNERDataset(processor, label_mapping, img_path, aux_path, rcnn_img_path, max_seq=cfg.max_seq,
                                  ignore_idx=cfg.ignore_idx, aux_size=cfg.aux_size, rcnn_size=cfg.rcnn_size,
                                  mode='train', cwd=cfg.cwd)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)

    dev_dataset = MMPNERDataset(processor, label_mapping, img_path, aux_path, rcnn_img_path, max_seq=cfg.max_seq,
                                ignore_idx=cfg.ignore_idx, aux_size=cfg.aux_size, rcnn_size=cfg.rcnn_size, mode='dev',
                                cwd=cfg.cwd)
    dev_dataloader = DataLoader(dev_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    test_dataset = MMPNERDataset(processor, label_mapping, img_path, aux_path, rcnn_img_path, max_seq=cfg.max_seq,
                                 ignore_idx=cfg.ignore_idx, aux_size=cfg.aux_size, rcnn_size=cfg.rcnn_size, mode='test',
                                 cwd=cfg.cwd)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = MCIPSCLCRFModel(LABEL_LIST, cfg)

    trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, model=model,
                      label_map=label_mapping, args=cfg, logger=logger, writer=writer)
    trainer.train()

@hydra.main(config_path="./conf", config_name='config.yaml')
def main(config):
    global cfg
    cfg = config
    study = optuna.create_study()  
    study.optimize(objective, n_trials=1)


if __name__ == '__main__':
    main()