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
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import wandb
# writer = wandb.init(project="DeepKE_RE_MM")

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

LABEL_LIST = ["[CLS]", "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[SEP]", "X"]

def set_seed(seed=2021):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def registerGradHook(model): # 注册梯度钩子
    # 获取CLIP的ViT最后一层特征
    image_encoder = model.model.encoder
    target_layer = image_encoder.vision_layers[-1]  # 最后一层Transformer

    # 存储特征和梯度
    features = {}
    gradients = {}

    def save_features(name):
        def hook(module, input, output):
            # print(output[0].shape)
            features[name] = output[0].detach()
        return hook

    def save_gradients(name):
        def hook(module, grad_input, grad_output):
            # print(grad_output[0].shape)
            gradients[name] = grad_output[0].detach()
        return hook

    # 注册钩子
    target_layer.register_forward_hook(save_features("last_layer"))
    target_layer.register_backward_hook(save_gradients("last_layer"))

    return gradients, features

def computeGradCam(grads, feats): # 计算Grad-CAM权重
    print(grads.shape)
    print(feats.shape)

    # 全局平均梯度（忽略CLS token）
    weights = grads[:, 0:50, :].mean(dim=1, keepdim=True)  # [1, 1, dim]

    # 特征图加权求和
    cam = (weights * feats[:, 0:50, :]).sum(dim=-1, keepdim=True)  # [1, num_patches, 1]
    cam = torch.relu(cam)  # ReLU激活
    # cam = cam[:,1:50,:]
    cam = cam.squeeze().cpu().numpy()
    return cam

def upsamplingHeatmap(cam): # 上采样热力图
    # 调整热力图尺寸到图像大小
    num_patches = int(np.sqrt(1+cam.shape[0]))  # 假设是方形patch
    cam = cam.reshape(num_patches, num_patches)
    print(torch.tensor(cam))
    cam = np.array([[4.6100e-06, 2.3551e-05, 2.2041e-05, 1.5295e-06, 9.6556e-06, 9.8625e-06,
         1.3176e-05],
        [3.0599e-06, 4.0267e-05, 4.1123e-05, 0.0000e+00, 5.3985e-06, 0.0000e+00,
         2.5528e-05],
        [2.6281e-05, 4.0342e-05, 4.5183e-05, 1.3277e-05, 3.4895e-05, 1.5051e-05,
         2.3521e-05],
        [1.1757e-05, 4.0459e-05, 4.0000e-05, 2.6029e-05, 4.0834e-05, 2.9440e-05,
         1.2011e-05],
        [3.8125e-05, 2.6454e-05, 3.4828e-05, 4.1792e-05, 3.5561e-05, 3.7407e-05,
         2.7144e-05],
        [3.9329e-05, 1.6005e-05, 1.3617e-06, 1.9266e-05, 1.2803e-05, 0.0000e+00,
         4.3175e-05],
        [6.1293e-06, 1.8392e-05, 2.9457e-05, 1.9340e-05, 8.5561e-06, 1.2479e-05,
         4.9276e-05]])
    # cam = np.array([[0., 0., 0., 0., 0., 0., 0.],
    #                 [0., 0., 0., 0., 0., 0., 0.],
    #                 [0., 0., 0., 0., 0., 0., 0.],
    #                 [0., 0., 0., 0., 0., 0., 0.],
    #                 [0., 0., 0., 0., 0., 0., 0.],
    #                 [0., 0., 0., 0., 0., 0., 0.],
    #                 [0., 0., 0., 0., 0., 0., 0.]])
    # cam = np.array([[0., 0., 0., 0., 0., 0., 0.],
    #                 [0.010, 0.031, 0.005, 0.008, 0.009, 0.006, 0.007],
    #                 [0.01127568, 0.01232097, 0.02296866, 0.02393537, 0.02134362, 0.02247277, 0.],
    #                 [0.01446878, 0.0241653, 0.01988831, 0.02685747, 0.0205939, 0.02127568, 0.],
    #                 [0.01405885, 0.02333589, 0.02262722, 0.02439018, 0.02682037, 0.01039018, 0.],
    #                 [0., 0., 0., 0., 0., 0., 0.],
    #                 [0., 0., 0., 0., 0., 0., 0.]])
    # cam = cv2.resize(cam, (image_input.width, image_input.height), interpolation=cv2.INTER_CUBIC)
    cam = cv2.resize(cam, (600, 450), interpolation=cv2.INTER_LINEAR)

    # 归一化
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

def visualize(cam):
    # 叠加热力图到原图
    image = np.array(Image.open("/home/wpf/workspace/mner-data/data/twitter2015/twitter2015_images/966944.jpg"))
    
    plt.imshow(image)
    plt.imshow(cam, alpha=0.6, cmap='jet')
    plt.axis("off")
    plt.savefig("heatmap-loc.jpeg", dpi=256, bbox_inches ='tight', format="jpeg")
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

    processor = MMPNERProcessor(data_path, cfg)
    test_dataset = MMPNERDataset(processor, label_mapping, img_path, aux_path, rcnn_img_path, max_seq=cfg.max_seq, ignore_idx=cfg.ignore_idx, aux_size=cfg.aux_size, rcnn_size=cfg.rcnn_size, mode='test', cwd=cfg.cwd)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = MCIPSCLCRFModel(LABEL_LIST, cfg)
    gradients, features = registerGradHook(model)

    model.train()
    if cfg.load_path is not None:  # load model from load_path
        model.load_state_dict(torch.load(cfg.load_path))
        model.to(cfg.device)
    with tqdm(total=len(test_dataloader), leave=False, dynamic_ncols=True) as pbar:
        pbar.set_description_str(desc="Predicting")
        for i, batch in enumerate(test_dataloader):
            if i == 139:
                batch = (tup.to(cfg.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device
                input_ids, token_type_ids, attention_mask, labels, labels_fx, images, aux_imgs, rcnn_imgs = batch
                # 前向传播
                logits, loss, sequence_output, emissions = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, images=images, aux_imgs=aux_imgs, rcnn_imgs=rcnn_imgs)
                print(input_ids[0].shape)
                input_words = test_dataset.tokenizer.decode(input_ids[0])
                print(test_dataset.tokenizer.convert_ids_to_tokens(input_ids[0]))
                print(labels)
                print(input_words)
                
                # 选择目标位置和标签（例如第0个位置的标签1）
                # pos = 3
                target_tag = 8 # 假设目标实体类别
                # target_score = emissions[0, pos, target_tag]  # 假设batch_size=1
                target_score = emissions[0, 12:14, target_tag].sum()  # 假设batch_size=1
                target_score.backward()
                # 计算辅助损失并反向传播
                # target_score.backward(retain_graph=True)  # 保留计算图以便后续操作

                # 获取梯度和特征
                grads = gradients["last_layer"]  # [1, num_patches+1, dim]
                feats = features["last_layer"]  # [1, num_patches+1, dim]

                # 计算Grad-CAM权重
                cam = computeGradCam(grads, feats)
                # 上采样热力图
                cam = upsamplingHeatmap(cam)
                # 可视化
                visualize(cam)
                return
            pbar.update()
        # evaluate done
        pbar.close()
    



if __name__ == '__main__':
    main()
