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
from evaluation.panopticbev_eval.meters import AverageMeter, ConfusionMatrixMeter
from evaluation.panopticbev_eval.conf_mat import confusion_matrix

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
        choices=["kitti360", "nuscenes"],
        help="dataset to evaluate on",
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
    dataset = opt.dataset

    config = OmegaConf.load(opt.config)
    ckpt_path = opt.ckpt
    model_config_path = os.path.join(os.getcwd(), ckpt_path, "configs")
    project_file = [f for f in os.listdir(model_config_path) if f.endswith("project.yaml")][0]
    model_config_path = os.path.join(model_config_path, project_file)
    model_config = OmegaConf.load(model_config_path).model
    model_config.params.ckpt_path = os.path.join(ckpt_path, "checkpoints", "last.ckpt")
    model = instantiate_from_config(model_config)
    
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

    #sem_conf = ConfusionMatrixMeter(n_classes)
    #sem_miou = AverageMeter(())
    sem_conf_mat = torch.zeros(n_classes, n_classes, dtype=torch.double)

    for i,batch in enumerate(tqdm(data._val_dataloader())):
        
        img = batch['image'].permute(0,3,1,2).float()
        bev = batch['bev'].long()
        cam = batch['cam'].float()


        if torch.cuda.is_available():
            img = img.cuda()
            bev = bev.cuda()
            cam = cam.cuda()
        
        with torch.no_grad():
            logits, _ = model(img, cam)
        
        #sample = batch["sample"] if "sample" in batch else None

        amax = torch.argmax(logits, dim =1, keepdim= True)
        
        bev = bev.squeeze()
        amax = amax.squeeze()

        conf_mat = confusion_matrix(n_classes, amax, bev)
        sem_conf_mat += conf_mat.cpu()

        #sem_conf.update(conf_mat.cpu())
    
    sem_conf_mat = sem_conf_mat.cpu()[:n_classes, :]
    sem_intersection = sem_conf_mat.diag()
    sem_union = ((sem_conf_mat.sum(dim=1) + sem_conf_mat.sum(dim=0)[:n_classes] - sem_conf_mat.diag()) + 1e-8)
    sem_miou = sem_intersection / sem_union


    os.makedirs(os.path.join(base_result_path), exist_ok=True)
    result_save_path = os.path.join(base_result_path, opt.savename + '.txt')
    with open(result_save_path, 'w') as file:
        for name, iou_score in zip(eval_config.class_names, sem_miou):
            file.write('\n{:20s} {:.4f}'.format(name, iou_score))
        file.write('\n{:20s} {:.4f}'.format('MEAN', sem_miou.mean()))

