import torch
import torch.nn as nn
import torch.nn.functional as F
from bev.data.nuscenes_utils import NUSCENES_PRIORS

class BCELoss(nn.Module):

    def __init__(self, priors=None):
        super().__init__()
        
        if priors is not None:
            assert priors in ['nuscenes']
            if priors == 'nuscenes':
                self.register_buffer('class_weights', torch.sqrt(1 / torch.Tensor(list(NUSCENES_PRIORS.values()))))
            #if torch.cuda.is_available():
            #    current_device = torch.cuda.current_device()
            #    self.class_weights = self.class_weights.to(current_device)
        else:
            self.class_weights = None
        
    def forward(self, prediction, target, mask, split):
        mask = mask.unsqueeze(1).float()
        if self.class_weights is not None:
            weights = (self.class_weights.to(prediction)[None, :, None, None] - 1) * target + 1.
            mask = weights * mask
        loss = F.binary_cross_entropy_with_logits(prediction,target, weight=mask)
        return loss, {"{}/loss".format(split): loss_combined.clone().detach().mean()}

class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


class CELoss(nn.Module):
    
    def __init__(self, ignore_index = 255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, prediction, target, mask, split):
        loss = F.cross_entropy(prediction, target.squeeze(1).long(), ignore_index=self.ignore_index, reduction='none')

        mask = mask.float() / 10000
        loss *= mask
        loss = loss.mean()

        return loss, {"{}/loss".format(split): loss_combined.clone().detach().mean()}

class CELossWithDistance(nn.Module):

    def __init__(self, extrinsics, out_shape = (200, 200), resolution = 0.25, ignore_index = 255):
        super().__init__()
        self.ignore_index = ignore_index
        
        rows = torch.arange(0, out_shape[0])
        cols = torch.arange(0, out_shape[1])
        rr, cc = torch.meshgrid(rows, cols)
        idx_mesh = torch.cat([rr.unsqueeze(0), cc.unsqueeze(0)], dim=0)
        ego_position = torch.tensor([out_shape[0] // 2, 0]).view(-1, 1, 1)
        pos_mesh = idx_mesh - ego_position
        X, Z = pos_mesh[0] * resolution, pos_mesh[1] * resolution
        self.Y = abs(float(extrinsics['translation'][2]))
        self.register_buffer("X", X)
        self.register_buffer("Z", Z)

    def forward(self, prediction, target, mask, intrinsics):
        
        loss = F.cross_entropy(prediction, target.squeeze(1).long(), ignore_index=self.ignore_index, reduction='none')

        f_x = intrinsics[:, 0, 0][:, None, None]
        f_y = intrinsics[:, 1, 1][:, None, None]

        
        S = torch.sqrt((f_x**2 * self.Z[None, :, :]**2) + (f_x*self.X[None, :, :] + f_y*self.Y)**2) / (self.Z[None, :, :]**2)
        sensitivity_map = 1 / torch.log(1 + S)
        sensitivity_map[torch.isnan(sensitivity_map)] = 0.
        # sensitivity_wt = sensitivity_map * 10
        sensitivity_wt = sensitivity_map * 10

        # Distance-based weighting
        loss *= (1 + sensitivity_wt)

        # Multiply the instace-based weights mask
        loss *= (mask.float() / 10000)

        loss = loss.mean()

        return loss


class BackgroundLoss(nn.Module):
    
    def __init__(self, background_classes, foreground_classes, ignore_index = 255):
        super().__init__()
        self.background_classes = background_classes
        self.foreground_classes = foreground_classes
        self.ignore_index = ignore_index

    def forward(self, prediction, target):

        target_mask = (target == self.ignore_index)
        background_logits = prediction[:, self.background_classes, :, :].sum(dim=1)
        foreground_logits = prediction[:, self.foreground_classes, :, :].sum(dim=1)
        logits = torch.stack([foreground_logits, background_logits], dim= 1)
        
        target_mask = target_mask.float()
        target = torch.cat([torch.zeros_like(target_mask), target_mask], dim = 1)
        loss = F.binary_cross_entropy_with_logits(logits, target, weight = target_mask)

        return loss

class VisibilityLoss(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, prediction, vis_mask):
        return F.binary_cross_entropy_with_logits(prediction, vis_mask)

class CombinedLoss(nn.Module):

    def __init__(self, extrinsics, background_classes, foreground_classes, out_shape = (200, 200), 
            resolution = 0.25, ignore_index = 255, loss_weights=[0.8, 0.1, 0.1]):
        super().__init__()
        
        self.celoss = CELossWithDistance(extrinsics, out_shape, resolution, ignore_index)
        self.backgroundloss = BackgroundLoss(background_classes, foreground_classes, ignore_index)
        self.visloss = VisibilityLoss()
        self.ignore_index = ignore_index
        self.loss_weights = loss_weights

    def forward(self, class_predict, vis_predict, target, mask, intrinsics, split):

        celoss = self.celoss(class_predict, target, mask, intrinsics)
        backgroundloss = self.backgroundloss(class_predict, target)
        vis_mask = (target == self.ignore_index).float()
        vis_loss = self.visloss(vis_predict, vis_mask)

        a, b, c = self.loss_weights
        loss_combined = a * celoss + b * backgroundloss + c * vis_loss

        return loss_combined, {"{}/total_loss".format(split): loss_combined.clone().detach().mean(),
                               "{}/seg_loss".format(split): celoss.detach().mean(),
                               "{}/background_loss".format(split): backgroundloss.detach().mean(),
                               "{}/visibility_loss".format(split): vis_loss.detach().mean()}
