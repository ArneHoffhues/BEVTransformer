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
        const=True,
        default="",
        nargs="?",
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
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
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

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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
    seed_everything(opt.seed)
    exp_name=opt.savename

    config = OmegaConf.load(opt.config) 
    model = instantiate_from_config(config.model)
    
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    
    eval_config = config.eval
    keys = np.arange(0, eval_config.n_classes)
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

    for i,batch in enumerate(tqdm(data._val_dataloader())):
        
        img = batch['image'].permute(0,3,1,2).float()
        bev = batch['bev'].permute(0,3,1,2).float()
        mask = batch['mask'].bool()
        cam = batch['cam'].float()

        if torch.cuda.is_available():
            img = img.cuda()
            bev = bev.cuda()
            mask = mask.cuda()
            cam = cam.cuda()
        
        with torch.no_grad():
            logits = model(img, cam)
        
        sample = batch["sample"] if "sample" in batch else None

        amax = torch.argmax(logits, dim =1, keepdim= True)
        one_hot = F.one_hot(amax, num_classes=eval_config.n_classes)
        one_hot = one_hot.transpose(1, -1).squeeze(-1)

        confusion.update(one_hot.bool(), bev.bool(), mask=mask)
        
        if sample is not None:
        #img = np.squeeze(np.argmax(logits.cpu().numpy().transpose((0, 2, 3, 1)), axis = 3)).astype(np.uint8)
            img = amax.squeeze().cpu().numpy().astype(np.uint8)
            seq_name, img_name = sample[0][0], sample[1][0]
            os.makedirs(os.path.join(base_inference_path, opt.savename, seq_name), exist_ok=True)
            pil_image = Image.fromarray(img, 'P')
            pil_image.putpalette(palette)
            save_path = os.path.join(base_inference_path, opt.savename, seq_name, img_name)
            pil_image.save(save_path)

    os.makedirs(os.path.join(base_result_path), exist_ok=True)
    result_save_path = os.path.join(base_result_path, opt.savename + '.txt')
    with open(result_save_path, 'w') as file:
        for name, iou_score in zip(eval_config.class_names, confusion.iou):
            file.write('\n{:20s} {:.3f}'.format(name, iou_score))
        file.write('\n{:20s} {:.3f}'.format('MEAN', confusion.mean_iou))

