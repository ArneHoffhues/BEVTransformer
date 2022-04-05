import os, math
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from bev.utils import instantiate_from_config
from evaluation.confusion_metric import ConfusionMatrixMetric


class VED(pl.LightningModule):
    def __init__(self,
                 net_config,
                 loss_config,
                 n_labels,
                 class_names=None,
                 colorize=None,
                 ckpt_path=None,
                 ignore_index = None,
                 ignore_keys=[],
                 rgb_key = 'image',
                 bev_key = 'bev',
                 mask_key = 'mask',
                 ):
        super().__init__()
        self.rgb_key = rgb_key
        self.bev_key = bev_key
        self.mask_key = mask_key
        self.class_names = class_names
        self.ignore_index = ignore_index
        self.n_labels = n_labels

        self.train_iou = ConfusionMatrixMetric(n_labels, class_names)
        self.val_iou = ConfusionMatrixMetric(n_labels, class_names)
        
        self.loss = instantiate_from_config(config=loss_config)
        self.net = instantiate_from_config(config=net_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if colorize is not None:
            if self.ignore_index is None:
                assert len(colorize) == self.n_labels
            else:
                assert len(colorize) == self.n_labels + 1
            colorize = torch.tensor(colorize, dtype=torch.float32).transpose(0,1).unsqueeze(2).unsqueeze(3).contiguous()
            colorize = colorize / 127.5 - 1.0
            self.register_buffer("colorize", colorize)

    def forward(self, x, is_training):
        x = self.net(x, is_training)
        return x


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


    def get_input(self, key, batch):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        x = x.float()
        return x

    def get_inputs(self, batch, N=None):
        x = self.get_input(self.rgb_key, batch)
        bev = self.get_input(self.bev_key, batch)
        mask = batch[self.mask_key]
        if N is not None:
            x = x[:N]
            bev = bev[:N]
            mask = mask[:N]
        return x, bev, mask

    def shared_step(self, batch, batch_idx, split):
        x, bev, mask = self.get_inputs(batch)
        logits, mu, logvar = self(x, split=='train')

        loss, loss_dict = self.loss(logits, mu, logvar, bev, split)

        pred = self.logits_to_one_hot(logits)
        bev = self.labels_to_one_hot(bev)

        if self.ignore_index is None:
            conf_mat_mask = mask.bool()
        else:
            bev, conf_mat_mask = bev[:, :-1, :, :], ~bev[:,-1, :, :].bool()

        return loss, loss_dict, pred, bev, conf_mat_mask

    def training_step(self, batch, batch_idx):
        loss, loss_dict, pred, bev, mask = self.shared_step(batch, batch_idx, 'train')
        self.train_iou.update(pred.bool(), bev.bool(), mask=mask)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict, pred, bev, mask = self.shared_step(batch, batch_idx, 'val')
        self.val_iou.update(pred.bool(), bev.bool(), mask=mask)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def training_epoch_end(self, outputs):
        train_ious = self.train_iou.compute('train')
        self.log_dict(train_ious)
        self.train_iou.reset()

    def validation_epoch_end(self, outputs):
        val_ious = self.val_iou.compute('val')
        self.log_dict(val_ious)
        self.val_iou.reset()

    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.Adam(list(self.net.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
    
    def log_images(self, batch, **kwargs):
        N = 4

        log = dict()
        x, bev_gt, mask = self.get_inputs(batch, N)
        x = x.to(self.device)
        bev, _, _ = self(x, False)
        #colorize
        assert bev.shape[1] == self.n_labels
        # convert logits to indices
        bev = torch.argmax(bev, dim=1, keepdim=True)
        if self.ignore_index is None:
            bev = F.one_hot(bev, num_classes=self.n_labels)
        else:
            bev = F.one_hot(bev, num_classes = self.n_labels + 1)
            ignore_region = bev_gt == self.ignore_index
            bev_non_hall = bev.clone().detach()
            bev_non_hall[:, :, :, :, -1][ignore_region] = 1
            bev_non_hall[:, :, :, :, :-1][ignore_region, :] = 0
            bev_non_hall = bev_non_hall.squeeze(1).permute(0, 3, 1, 2).float()
            bev_non_hall = self.to_rgb(bev_non_hall)
            log["bev_not_hallucinated"] = bev_non_hall
        bev = bev.squeeze(1).permute(0, 3, 1, 2).float()
        bev = self.to_rgb(bev)
        log["inputs"] = x
        if self.ignore_index is not None:
            bev_gt = self.labels_to_one_hot(bev_gt).float()
        log["bev_gt"] = self.to_rgb(bev_gt)
        log["bev_hallucinated"] = bev
        return log

    def logits_to_one_hot(self, x):
        x = torch.argmax(x, dim=1, keepdim=True)
        x = F.one_hot(x, num_classes=self.n_labels)
        x = x.squeeze(1).permute(0, 3, 1, 2)
        return x

    def labels_to_one_hot(self, x):
        if self.ignore_index is None:
            return x
        else:
            x = x.long()
            x[x == self.ignore_index] = self.n_labels
            x = F.one_hot(x, num_classes=self.n_labels + 1)
            x = x.squeeze(1).permute(0, 3, 1, 2)
            return x

    def to_rgb(self, x):
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


