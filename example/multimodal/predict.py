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

DATA_PATH = {
    'twitter15':{
        'test': 'data/twitter2015/test.txt',
        'test_auximgs': 'data/twitter2015/twitter2015_test_dict.pth',
        'rcnn_img_path': 'data/twitter2015',
        'img2crop': 'data/twitter2015/twitter15_detect/twitter15_img2crop.pth'},
    'twitter17':{
        'test': 'data/twitter2017/test.txt',
        'test_auximgs': 'data/twitter2017/twitter2017_test_dict.pth',
        'rcnn_img_path': 'data/twitter2017',
        'img2crop': 'data/twitter2017/twitter17_detect/twitter17_img2crop.pth'}
}

IMG_PATH = {
    'twitter15': 'data/twitter2015/twitter2015_images',
    'twitter17': 'data/twitter2017/twitter2017_images'
}

AUX_PATH = {
    'twitter15': {'test': 'data/twitter2015/twitter2015_aux_images/test/crops'},

    'twitter17': {'test': 'data/twitter2017/twitter2017_aux_images/test/crops'}
}

# LABEL_LIST = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
LABEL_LIST = ["[CLS]","O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[SEP]", "X" ]

def set_seed(seed=2021):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def plotTSNE_2D(token_vecs, entity_labels):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import LabelEncoder
    import umap

    for i in range(len(entity_labels)):
        label = entity_labels[i]
        if label.startswith("B-") or label.startswith("I-"):  # 只取实体token
            entity_type = label[2:]  # 去掉 B-/I- 前缀
            entity_labels[i] = entity_type

    # 转为numpy格式
    token_vecs = np.array(token_vecs)
    entity_labels = np.array(entity_labels)

    # 使用LabelEncoder编码标签为整数，便于绘图着色
    label_encoder = LabelEncoder()
    label_ids = label_encoder.fit_transform(entity_labels)

    # 使用TSNE降维
    reducer = TSNE(n_components=2, perplexity=100, random_state=42, max_iter=1000)
    # reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=2021)
    reduced_vecs = reducer.fit_transform(token_vecs)

    # 可视化
    plt.figure(figsize=(12, 10))
    for label_id in np.unique(label_ids):
        idx = label_ids == label_id
        plt.scatter(reduced_vecs[idx, 0], reduced_vecs[idx, 1], label=label_encoder.inverse_transform([label_id])[0], alpha=0.7)

    plt.title("t-SNE 3D of Entity Token Vectors")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.savefig("tsne-twitter-2015.eps", dpi=256, bbox_inches ='tight', format="eps")
    plt.show()

def plotTSNE_3D(token_vecs, entity_labels):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import LabelEncoder
    import umap

    for i in range(len(entity_labels)):
        label = entity_labels[i]
        if label.startswith("B-") or label.startswith("I-"):  # 只取实体token
            entity_type = label[2:]  # 去掉 B-/I- 前缀
            entity_labels[i] = entity_type

    # 转为numpy格式
    token_vecs = np.array(token_vecs)
    entity_labels = np.array(entity_labels)

    # 使用LabelEncoder编码标签为整数，便于绘图着色
    label_encoder = LabelEncoder()
    label_ids = label_encoder.fit_transform(entity_labels)

    # 使用TSNE降维
    reducer = TSNE(n_components=3, perplexity=30, random_state=2021, max_iter=5000)
    # reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=2021)
    reduced_vecs = reducer.fit_transform(token_vecs)

    # 可视化
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_axes([0,0.01,1,0.99], projection='3d')
    for label_id in np.unique(label_ids):
        idx = label_ids == label_id
        # plt.scatter(reduced_vecs[idx, 0], reduced_vecs[idx, 1], label=label_encoder.inverse_transform([label_id])[0], alpha=0.7)
        ax.scatter(reduced_vecs[idx, 0], reduced_vecs[idx, 1], reduced_vecs[idx, 2], label=label_encoder.inverse_transform([label_id])[0], s = 10, alpha=0.7)

    fig.text(0.5, 0.01, "(b) T-SNE 3D of Entity Token Vectors on Twitter-2017", ha='center', fontsize=20)
    ax.set_xlabel("X", fontsize=20)
    ax.set_ylabel("Y", fontsize=20)
    ax.set_zlabel("Z", fontsize=20)

    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='z', labelsize=18)
    ax.legend(fontsize=18)
    plt.grid(True)
    plt.savefig("tsne-twitter-2017.eps", dpi=256, bbox_inches ='tight', format="eps")
    plt.show()


@hydra.main(config_path="./conf", config_name='config.yaml')
def main(cfg):
    # cwd = utils.get_original_cwd()
    # cfg.cwd = cwd
    print(cfg)

    if cfg.use_wandb:
        writer = wandb.init(project="DeepKE_NER_MM")
    else:
        writer=None

    set_seed(cfg.seed) # set seed, default is 1
    if cfg.save_path is not None:  # make save_path dir
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path, exist_ok=True)
    print(cfg)

    label_mapping = {label:idx for idx, label in enumerate(LABEL_LIST, 0)}
    # label_mapping["PAD"] = 0
    data_path, img_path, aux_path = DATA_PATH[cfg.dataset_name], IMG_PATH[cfg.dataset_name], AUX_PATH[cfg.dataset_name]
    rcnn_img_path = DATA_PATH[cfg.dataset_name]['rcnn_img_path']
    rcnn_img_path=None
    processor = MMPNERProcessor(data_path, cfg)
    test_dataset = MMPNERDataset(processor, label_mapping, img_path, aux_path, rcnn_img_path, max_seq=cfg.max_seq, ignore_idx=cfg.ignore_idx, aux_size=cfg.aux_size, rcnn_size=cfg.rcnn_size, mode='test', cwd=cfg.cwd)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = MCIPSCLCRFModel(LABEL_LIST, cfg)

    trainer = Trainer(train_data=None, dev_data=None, test_data=test_dataloader, model=model, label_map=label_mapping, args=cfg, logger=logger, writer=writer)
    token_vercs, entity_labels = trainer.predict()

    plotTSNE_3D(token_vercs, entity_labels)


if __name__ == '__main__':
    main()
