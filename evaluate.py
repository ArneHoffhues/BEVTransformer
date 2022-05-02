import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only

from bev.model.model import BEVTransformer
from bev.data.data_module import DataModuleFromConfig, ValModuleFromConfig
from evaluation.confusion import BinaryConfusionMatrix
from tqdm import tqdm

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--savename",
        type=str,
        help="name for save directory",
        required=True,
    )
    parser.add_argument(
        "--config",
        nargs="?",
        metavar="base_config.yaml",
        help="paths to config",
        default=str,
        required=True,
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="path to checkpoint",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="kitti",
        choices=["kitti", "kitti360"],
        help="dataset to evaluate on",
    )
    parser.add_argument(
        "-d",
        "--depth",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="usage of depth",
    )
    
    return parser

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

# Create color palette from color dictionary
def palette_from_dict(c_dict):
    palette = []
    for i in np.arange(256):
        if i in c_dict:
            palette.extend(c_dict[i])
        else:
            palette.extend([0, 0, 0])
    return palette# Create color palette from color dictionary

if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    exp_name=opt.savename
    with_depth = opt.depth
    dataset = opt.dataset

    config = OmegaConf.load(opt.config)
    ckpt_path = opt.ckpt
    model_config_path = os.path.join(os.getcwd(), ckpt_path, "configs")
    project_file = [f for f in os.listdir(model_config_path) if f.endswith("project.yaml")][0]
    model_config_path = os.path.join(model_config_path, project_file)
    model_config = OmegaConf.load(model_config_path).model
    model_config.params.ckpt_path = os.path.join(ckpt_path, "checkpoints", "last.ckpt")
    if os.path.getsize(model_config.params.ckpt_path) == 0:
        ckpts = os.listdir(os.path.join(ckpt_path, "checkpoints"))
        epoch_ckpt = [ckpt for ckpt in ckpts if ckpt.startswith('epoch=')][0]
        ckpt = os.path.join(ckpt_path, "checkpoints", epoch_ckpt)
        model_config.params.ckpt_path = ckpt
    model = instantiate_from_config(model_config)
    
    if with_depth:
        config.data.params.validation.params.with_depth = True
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    
    eval_config = config.eval
    n_classes = eval_config.n_classes
    keys = np.arange(0, n_classes)
    values = eval_config.colorize
    colors_dict = dict(zip(keys, values))
    palette = palette_from_dict(colors_dict)
    base_inference_path = eval_config.inference_save_directory
    base_result_path = eval_config.result_save_directory

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()

    # Initialise confusion matrix
    confusion = BinaryConfusionMatrix(eval_config.n_classes)
    
    os.makedirs(os.path.join(base_inference_path, opt.savename, 'hallucinated'), exist_ok=True)
    os.makedirs(os.path.join(base_inference_path, opt.savename, 'not_hallucinated'), exist_ok=True)

    for i,batch in enumerate(tqdm(data._val_dataloader())):
        
        img = batch['image'].permute(0,3,1,2).float()
        if dataset == 'kitti':
            bev = batch['bev'][:, :, :, 1:].permute(0,3,1,2).long()
            mask = batch['mask'].bool()
        else:
            bev = batch['bev'].unsqueeze(1).long()
            mask = (bev != 255).bool()
            bev[bev == 255] = n_classes
            bev = F.one_hot(bev, num_classes=n_classes + 1)
            bev = bev.squeeze(1).permute(0, 3, 1, 2)
            bev = bev[:, :-1, :, :]
        cam = batch['cam'].float()

        if with_depth:
            depth = batch['depth'].unsqueeze(1).float()

        if torch.cuda.is_available():
            img = img.cuda()
            bev = bev.cuda()
            mask = mask.cuda()
            cam = cam.cuda()
            
            if with_depth:
                depth = depth.cuda()
        
        with torch.no_grad():
            if with_depth:
                logits, _ = model(img, cam, depth)
            else:
                logits, _ = model(img, cam)
        
        #sample = batch["sample"] if "sample" in batch else None

        amax = torch.argmax(logits, dim =1, keepdim= True)
        one_hot = F.one_hot(amax, num_classes=eval_config.n_classes)
        one_hot = one_hot.transpose(1, -1).squeeze(-1)

        confusion.update(one_hot.bool(), bev.bool(), mask=mask)
        
        if dataset == 'kitti':
            seq_name, img_name  = data.datasets['validation'].samples[i]
        elif dataset == 'kitti360':
            seq_name = data.datasets['validation'].split
            img_name = data.datasets['validation'].images[i]['id'] + '.png'
        else:
            raise ValueError(f'unknown dataset name: {dataset}')

        #img = np.squeeze(np.argmax(logits.cpu().numpy().transpose((0, 2, 3, 1)), axis = 3)).astype(np.uint8)
        img = amax.squeeze().cpu().numpy().astype(np.uint8)
        os.makedirs(os.path.join(base_inference_path, opt.savename, 'hallucinated', seq_name), exist_ok=True)
        hal_image = Image.fromarray(img, 'P')
        hal_image.putpalette(palette)
        save_path = os.path.join(base_inference_path, opt.savename, 'hallucinated', seq_name, img_name)
        hal_image.save(save_path)
        os.makedirs(os.path.join(base_inference_path, opt.savename, 'not_hallucinated', seq_name), exist_ok=True)
        mask = mask.squeeze().cpu().numpy().astype(bool)
        img[~mask] = 255
        non_hal_image = Image.fromarray(img, 'P')
        non_hal_image.putpalette(palette)
        save_path = os.path.join(base_inference_path, opt.savename, 'not_hallucinated', seq_name, img_name)
        non_hal_image.save(save_path)


    os.makedirs(os.path.join(base_result_path), exist_ok=True)
    result_save_path = os.path.join(base_result_path, opt.savename + '.txt')
    with open(result_save_path, 'w') as file:
        for name, iou_score in zip(eval_config.class_names, confusion.iou):
            file.write('\n{:20s} {:.3f}'.format(name, iou_score))
        file.write('\n{:20s} {:.3f}'.format('MEAN', confusion.mean_iou))

