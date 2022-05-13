import os, math
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.profiler import profile, record_function, ProfilerActivity

from bev.utils import instantiate_from_config
from evaluation.confusion import BinaryConfusionMatrix
from evaluation.confusion_metric import ConfusionMatrixMetric
from bev.model.loss import CombinedLoss


class BEVTransformer(pl.LightningModule):
    def __init__(self,
                 loss_config,
                 n_labels,
                 n_embed,
                 class_names=None,
                 pos_encoding_config = None,
                 height_pos_encoding_config = None,
                 transformer_config=None,
                 backbone_config=None,
                 net3d_config=None,
                 decoder_config = None,
                 colorize=None,
                 ckpt_path=None,
                 ignore_index = None,
                 mask_ignore = False,
                 score_threshold = None,
                 ignore_keys=[],
                 rgb_key = 'image',
                 bev_key = 'bev',
                 cam_key = 'cam',
                 mask_key = 'mask',
                 ):
        super().__init__()
        self.rgb_key = rgb_key
        self.bev_key = bev_key
        self.cam_key = cam_key
        self.mask_key = mask_key
        self.n_labels = n_labels
        self.class_names = class_names
        self.ignore_index = ignore_index
        self.mask_ignore = mask_ignore
        self.score_threshold = score_threshold
         
        #self.train_conf_mat = BinaryConfusionMatrix(n_labels)
        #self.val_conf_mat = BinaryConfusionMatrix(n_labels)
        self.train_iou = ConfusionMatrixMetric(n_labels, class_names)
        self.val_iou = ConfusionMatrixMetric(n_labels, class_names)

        if pos_encoding_config is not None:
            self.pos_encoding = instantiate_from_config(config=pos_encoding_config)
        else:
            self.pos_encoding = None

        if height_pos_encoding_config is not None:
            self.height_pos_encoding = instantiate_from_config(config=height_pos_encoding_config)
        else:
            self.height_pos_encoding = None

        if transformer_config is not None:
            self.transformer = instantiate_from_config(config=transformer_config)

        if backbone_config is not None:
            self.backbone = instantiate_from_config(config=backbone_config)
        
        if net3d_config is not None:
            self.net3d = instantiate_from_config(config=net3d_config)
        
        if decoder_config is not None:
            self.decoder = instantiate_from_config(config=decoder_config)
            #self.decoder = nn.Conv2d(n_embed, n_labels, kernel_size= 1)
        self.loss = instantiate_from_config(config=loss_config)
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if colorize is not None:
            if self.ignore_index is None and not mask_ignore:
                assert len(colorize) == self.n_labels
            else:
                assert len(colorize) == self.n_labels + 1
            colorize = torch.tensor(colorize, dtype=torch.float32).transpose(0,1).unsqueeze(2).unsqueeze(3).contiguous()
            colorize = colorize / 127.5 - 1.0
            self.register_buffer("colorize", colorize)
    
    def forward(self, x, cam):
        f = self.backbone(x)
        f_3d, init = self.net3d(f, cam)
        init_pos = self.pos_encoding(init) if self.pos_encoding else None
        height_pos = self.height_pos_encoding(f_3d) if self.height_pos_encoding else None
        out = self.transformer(f_3d, init, init_pos, height_pos)
        out = self.decoder(out)

        return out


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
        cam = batch[self.cam_key].float()
        mask = batch[self.mask_key]
        if N is not None:
            x = x[:N]
            bev = bev[:N]
            cam = cam[:N]
            mask = mask[:N]
        return x, bev, cam, mask

    def shared_step(self, batch, batch_idx, split):
        x, bev, cam, mask = self.get_inputs(batch)
        logits, vis_logits = self(x, cam)

        if isinstance(self.loss, CombinedLoss):
            loss, loss_dict = self.loss(logits, vis_logits, bev, mask, cam, split)
        else:
            loss, loss_dict = self.loss(logits, bev, mask, split)

        #loss, loss_dict = self.loss(logits, bev, mask, split)
        
        if self.score_threshold:
            scores = logits.sigmoid()
            pred = scores > self.score_threshold
        else:
            pred = self.logits_to_one_hot(logits)
        bev = self.labels_to_one_hot(bev)
        
        if self.ignore_index is None:
            conf_mat_mask = mask.bool()
        else:
            bev, conf_mat_mask = bev[:, :-1, :, :], ~bev[:,-1, :, :].bool() 
        
        return loss, loss_dict, pred, bev, conf_mat_mask

    def training_step(self, batch, batch_idx):
        loss, loss_dict, pred, bev, mask = self.shared_step(batch, batch_idx, 'train')
        #self.train_conf_mat.update(pred.bool(), bev.bool(), mask=mask)      
        self.train_iou.update(pred.bool(), bev.bool(), mask=mask)

        #if self.class_names is not None:
        #    self.log_IoUs(self.train_conf_mat, 'train')
        #else:
        #    self.log("train/IoU", self.train_conf_mat.mean_iou, on_step = False, on_epoch=True)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict, pred, bev, mask = self.shared_step(batch, batch_idx, 'val')
        #self.val_conf_mat.update(pred.bool(), bev.bool(), mask=mask)
        self.val_iou.update(pred.bool(), bev.bool(), mask=mask)

        #if self.class_names is not None:
        #    self.log_IoUs(self.val_conf_mat, 'val')
        #else:
        #    self.log("val/IoU", self.val_conf_mat.mean_iou, on_step = False, on_epoch=True)
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def log_IoUs(self, confusion_matrix, split):
        for name, iou_score in zip(self.class_names, confusion_matrix.iou):
            self.log(f'{split}/IoU/{name}', iou_score, on_step=False, on_epoch=True)
        self.log(f'{split}/IoU/MEAN', confusion_matrix.mean_iou, on_step=False, on_epoch=True)

    #def on_train_epoch_start(self):
    #    self.train_conf_mat = BinaryConfusionMatrix(self.n_labels)

    #def on_validation_epoch_start(self):
    #    self.val_conf_mat = BinaryConfusionMatrix(self.n_labels)
    
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
        opt = torch.optim.Adam(list(self.transformer.parameters())+
                                  list(self.backbone.parameters())+
                                  list(self.net3d.parameters()) +
                                  list(self.decoder.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt
    
    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        N = 4

        log = dict()
        x, bev_gt, cam, mask = self.get_inputs(batch, N)
        x = x.to(self.device)
        cam = cam.to(self.device)
        bev, vis_prediction = self(x, cam)
        #colorize
        assert bev.shape[1] == self.n_labels
        # convert logits to indices
        bev = torch.argmax(bev, dim=1, keepdim=True)
        if self.ignore_index:
            bev = F.one_hot(bev, num_classes = self.n_labels + 1)
            ignore_region = bev_gt == self.ignore_index
            bev_non_hall = bev.clone().detach()
            bev_non_hall[:, :, :, :, -1][ignore_region] = 1
            bev_non_hall[:, :, :, :, :-1][ignore_region, :] = 0
            bev_non_hall = bev_non_hall.squeeze(1).permute(0, 3, 1, 2).float()
            bev_non_hall = self.to_rgb(bev_non_hall)
            log["bev_not_hallucinated"] = bev_non_hall
        elif self.mask_ignore:
            mask = mask.unsqueeze(1).bool()
            mask = ~mask
            bev = F.one_hot(bev, num_classes = self.n_labels + 1)
            bev_non_hall = bev.clone().detach()
            bev_non_hall[:, :, :, :, -1][mask] = 1
            bev_non_hall[:, :, :, :, :-1][mask] = 0
            bev_non_hall = bev_non_hall.squeeze(1).permute(0, 3, 1, 2).float()
            bev_non_hall = self.to_rgb(bev_non_hall)
            log["bev_not_hallucinated"] = bev_non_hall
        else:
            bev = F.one_hot(bev, num_classes=self.n_labels)
        bev = bev.squeeze(1).permute(0, 3, 1, 2).float()
        bev = self.to_rgb(bev)
        log["inputs"] = x
        if self.ignore_index is not None:
            bev_gt = self.labels_to_one_hot(bev_gt).float()
        if self.mask_ignore:
            B, C, H, W = bev_gt.shape
            unlabeled = torch.zeros((B, 1, H, W), device = x.device, dtype=bool)
            for i in reversed(range(bev_gt.shape[1])):
                mask = (bev_gt[:, i, :, :] == 1).unsqueeze(1)
                unlabeled[mask] = 1
                zero_mask = mask.repeat(1, i, 1, 1)
                bev_gt[:, :i, :, :][zero_mask] = 0
            unlabeled = (~unlabeled).float()
            bev_gt = torch.cat((bev_gt, unlabeled), 1)
        log["bev_gt"] = self.to_rgb(bev_gt)
        log["bev_hallucinated"] = bev
        return log

#    @torch.no_grad()
#    def log_images(self, batch, **kwargs):
#        N = 4

#        log = dict()
#        x, bev_gt, cam, mask = self.get_inputs(batch, N)
#        x = x.to(self.device)
#        cam = cam.to(self.device)
#        bev = self(x, cam)
#        #colorize
#        assert bev.shape[1] == self.n_labels
        # convert logits to indices
#        bev = torch.argmax(bev, dim=1, keepdim=True)
#        if self.ignore_index is None:
#            bev = F.one_hot(bev, num_classes=self.n_labels)
#        else:
#            bev = F.one_hot(bev, num_classes = self.n_labels + 1)
#        bev = bev.squeeze(1).permute(0, 3, 1, 2).float()
#        bev = self.to_rgb(bev)
#        log["inputs"] = x
#        if self.ignore_index is not None:
#            bev_gt = self.labels_to_one_hot(bev_gt).float()
#        log["bev_gt"] = self.to_rgb(bev_gt)
#        log["bev_output"] = bev
#        return log
    
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

class BEVTransformerWithDepth(BEVTransformer):
    def __init__(self,
                 loss_config,
                 n_labels,
                 n_embed,
                 class_names=None,
                 pos_encoding_config = None,
                 height_pos_encoding_config = None,
                 transformer_config=None,
                 backbone_config=None,
                 net3d_config=None,
                 decoder_config = None,
                 colorize=None,
                 ckpt_path=None,
                 ignore_index = None,
                 rgb_key = 'image',
                 bev_key = 'bev',
                 cam_key = 'cam',
                 mask_key = 'mask',
                 depth_key = 'depth'
                 ):
        super().__init__(loss_config=loss_config, n_labels=n_labels, n_embed=n_embed, class_names=class_names, pos_encoding_config=pos_encoding_config,
                height_pos_encoding_config=height_pos_encoding_config, transformer_config=transformer_config, backbone_config=backbone_config, net3d_config=net3d_config,
                decoder_config=decoder_config, colorize=colorize, ckpt_path=ckpt_path, ignore_index=ignore_index, rgb_key=rgb_key, cam_key=cam_key, mask_key=mask_key)
        self.depth_key = depth_key
        
    def get_inputs(self, batch, N=None):
        x = self.get_input(self.rgb_key, batch)
        bev = self.get_input(self.bev_key, batch)
        depth = self.get_input(self.depth_key, batch)
        cam = batch[self.cam_key].float()
        mask = batch[self.mask_key]
        if N is not None:
            x = x[:N]
            bev = bev[:N]
            cam = cam[:N]
            mask = mask[:N]
            depth = depth[:N]
        return x, bev, cam, mask, depth
    
    def forward(self, x, cam, depth):
        f = self.backbone(x)
        f_3d, init = self.net3d(f, cam, depth)
        init_pos = self.pos_encoding(init) if self.pos_encoding else None
        height_pos = self.height_pos_encoding(f_3d) if self.height_pos_encoding else None
        out = self.transformer(f_3d, init, init_pos, height_pos)
        out = self.decoder(out)

        return out

    def shared_step(self, batch, batch_idx, split):
        x, bev, cam, mask, depth = self.get_inputs(batch)
        logits, vis_logits = self(x, cam, depth)

        if isinstance(self.loss, CombinedLoss):
            loss, loss_dict = self.loss(logits, vis_logits, bev, mask, cam, split)
        else:
            loss, loss_dict = self.loss(logits, bev, mask, split)

        #loss, loss_dict = self.loss(logits, bev, mask, split)

        pred = self.logits_to_one_hot(logits)
        bev = self.labels_to_one_hot(bev)

        if self.ignore_index is None:
            conf_mat_mask = mask.bool()
        else:
            bev, conf_mat_mask = bev[:, :-1, :, :], ~bev[:,-1, :, :].bool()

        return loss, loss_dict, pred, bev, conf_mat_mask
    
    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        N = 4

        log = dict()
        x, bev_gt, cam, mask, depth = self.get_inputs(batch, N)
        x = x.to(self.device)
        cam = cam.to(self.device)
        depth = depth.to(self.device)
        bev, vis_prediction = self(x, cam, depth)
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

class NoTransformer(BEVTransformer):
    def __init__(self,
                 transformer_config,
                 backbone_config,
                 net3d_config,
                 loss_config,
                 n_labels,
                 n_embed,
                 colorize=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 rgb_key = 'image',
                 bev_key = 'bev',
                 cam_key = 'cam',
                 mask_key = 'mask',
                 ):
        super().__init__(transformer_config=transformer_config, backbone_config=backbone_config, net3d_config=net3d_config, loss_config=loss_config, n_labels=n_labels, 
                n_embed=n_embed, colorize=colorize, ckpt_path=ckpt_path, ignore_keys=ignore_keys, rgb_key=rgb_key, bev_key=bev_key, cam_key=cam_key, mask_key=mask_key)


    def forward(self, x, cam):
        f = self.backbone(x)
        f_3d = self.net3d(f, cam)
        collapsed = torch.mean(f_3d, 4)
        out = self.decoder(collapsed)

        return out

    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.Adam(list(self.backbone.parameters())+
                                  list(self.net3d.parameters()) +
                                  list(self.decoder.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt

class DepthAndSegUpsampler(BEVTransformer):
    def __init__(self,
                 sfsegnet_config,
                 depth_module_config,
                 decoder_config,
                 loss_config,
                 n_labels,
                 n_embed,
                 transformer_config=None,
                 pos_encoding_config=None,
                 class_names=None,
                 ignore_index=None,
                 colorize=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 rgb_key = 'image',
                 bev_key = 'bev',
                 depth_key = 'depth',
                 cam_key = 'cam',
                 mask_key = 'mask',
                 ):
        super().__init__(loss_config = loss_config, transformer_config=transformer_config, pos_encoding_config=pos_encoding_config,
                n_labels = n_labels, n_embed = n_embed, ignore_index = ignore_index, colorize = colorize, ckpt_path = ckpt_path, class_names = class_names)
        self.depth_key = depth_key
        
        self.seg_network = instantiate_from_config(config = sfsegnet_config)
        self.depth_projection = instantiate_from_config(config = depth_module_config)
        self.decoder = instantiate_from_config(config = decoder_config)

    def forward(self, x, cam, depth):
        seg = self.seg_network(x)
        projected_seg, _  = self.depth_projection(seg, depth, cam)
        logits, vis_logits = self.decoder(projected_seg)
        return logits, vis_logits
            
    def configure_optimizers(self):
        lr = self.learning_rate

        opt = torch.optim.Adam(list(self.seg_network.parameters()) +
                               list(self.depth_projection.parameters()) + 
                               list(self.decoder.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt
    
    def get_inputs(self, batch, N=None):
        x = self.get_input(self.rgb_key, batch)
        bev = self.get_input(self.bev_key, batch)
        depth = self.get_input(self.depth_key, batch)
        cam = batch[self.cam_key].float()
        mask = batch[self.mask_key]
        if N is not None:
            x = x[:N]
            bev = bev[:N]
            depth = depth[:N]
            cam = cam[:N]
            mask = mask[:N]
        return x, bev, depth, cam, mask
   
    def shared_step(self, batch, batch_idx, split):
        x, bev, depth, cam, mask = self.get_inputs(batch)
        logits, vis_logits = self(x, cam, depth)
        
        if isinstance(self.loss, CombinedLoss):
            loss, loss_dict = self.loss(logits, vis_logits, bev, mask, cam, split)
        else:
            loss, loss_dict = self.loss(logits, bev, mask, split)

        pred = self.logits_to_one_hot(logits)
        bev = self.labels_to_one_hot(bev)

        if self.ignore_index is None:
            conf_mat_mask = mask.bool()
        else:
            bev, conf_mat_mask = bev[:, :-1, :, :], ~bev[:,-1, :, :].bool()

        return loss, loss_dict, pred, bev, conf_mat_mask
    
    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        N = 4

        log = dict()
        x, bev_gt, depth, cam, mask = self.get_inputs(batch, N)
        x = x.to(self.device)
        cam = cam.to(self.device)
        depth = depth.to(self.device)
        bev, vis_prediction = self(x, cam, depth)
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
        log["depth"] = depth
        log["bev_hallucinated"] = bev
        return log

class DepthAndSegTransformer(DepthAndSegUpsampler):
    
    def __init__(self,
                 sfsegnet_config,
                 depth_module_config,
                 decoder_config,
                 loss_config,
                 n_labels,
                 n_embed,
                 transformer_config,
                 pos_encoding_config,
                 density_pos_encoding_config,
                 class_names=None,
                 ignore_index=None,
                 colorize=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 rgb_key = 'image',
                 bev_key = 'bev',
                 depth_key = 'depth',
                 cam_key = 'cam',
                 mask_key = 'mask',
                 ):
        super().__init__(sfsegnet_config=sfsegnet_config, depth_module_config=depth_module_config, decoder_config=decoder_config, loss_config = loss_config, 
                transformer_config=transformer_config, pos_encoding_config=pos_encoding_config, n_labels = n_labels, n_embed = n_embed, ignore_index = ignore_index, 
                colorize = colorize, ckpt_path = ckpt_path, class_names = class_names)
        self.density_pos_encoding = instantiate_from_config(config = density_pos_encoding_config)
        
    def forward(self, x, cam, depth):
        seg = self.seg_network(x)
        projected_seg, density  = self.depth_projection(seg, depth, cam)
        seg_pos = self.pos_encoding(projected_seg)
        density_pos = self.density_pos_encoding(density)
        seg, density = self.transformer(projected_seg, density, seg_pos, density_pos)
        logits, vis_logits = self.decoder(seg, density)
        return logits, vis_logits
    
    def configure_optimizers(self):
        lr = self.learning_rate

        opt = torch.optim.Adam(list(self.seg_network.parameters()) +
                               list(self.depth_projection.parameters()) +
                               list(self.transformer.parameters()) +
                               list(self.decoder.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt

