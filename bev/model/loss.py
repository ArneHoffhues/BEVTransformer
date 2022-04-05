import torch
import torch.nn as nn
import torch.nn.functional as F
from bev.data.nuscenes_utils import NUSCENES_PRIORS
from bev.data.kitti360_utils import KITTI360_PRIORS
from bev.data.nuscenes_bev_utils import NUSCENES_BEV_PRIORS

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


class VEDLoss(nn.Module):

    def __init__(self, ignore_index = 255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, prediction, mu, logvar, target, split):
        ce_loss = F.cross_entropy(prediction, target.squeeze(1).long(), ignore_index=self.ignore_index)

        kld = -0.5 *torch.mean(1 + logvar - mu.pow(2) -logvar.exp())
        loss = 0.9 * ce_loss + 0.1 * kld    
    
        return loss, {"{}/total_loss".format(split): loss.clone().detach().mean(),
                      "{}/seg_loss".format(split): ce_loss.detach().mean(),
                      "{}/kld_loss".format(split): kld.detach().mean(),}


class CELossWithDistance(nn.Module):

    def __init__(self, extrinsics, out_shape = (200, 200), resolution = 0.25, ignore_index = 255, priors=None):
        super().__init__()
        self.ignore_index = ignore_index

        if priors is not None:
            if priors == 'kitti360':
                kitti_priors = list(KITTI360_PRIORS.values())
                del kitti_priors[-1]
                del kitti_priors[7]
                self.register_buffer('class_weights', torch.sqrt(1 / torch.Tensor(kitti_priors)))
            elif priors == 'nuscenes':
                nuscenes_bev_priors = list(NUSCENES_BEV_PRIORS.values())
                del nuscenes_priors[-1]
                del nuscenes_priors[6]
                self.register_buffer('class_weights', torch.sqrt(1 / torch.Tensor(nuscenes_priors)))
            else:
                raise ValueError(f"Unknown prior option '{priors}'")
        
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
        
        if hasattr(self, 'class_weights'):
            loss = F.cross_entropy(prediction, target.squeeze(1).long(), ignore_index=self.ignore_index, weight = self.class_weights, reduction='none')
        else:
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
        if not hasattr(self, 'class_weights'):
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
        #probabilities = torch.exp(prediction - torch.logsumexp(prediction, dim = 1, keepdim = True))
        #background_prob = probabilities[:, self.background_classes, :, :].sum(dim=1)
        #foreground_prob = probabilities[:, self.foreground_classes, :, :].sum(dim=1)
        foreground_logits = prediction[:, self.foreground_classes, :, :]
        target_probs = torch.zeros_like(foreground_logits)
        #probs = torch.stack([foreground_prob, background_prob], dim= 1)

        target = target_mask.float()
        #target = torch.cat([torch.zeros_like(target_mask), target_mask], dim = 1)
        loss = F.binary_cross_entropy_with_logits(foreground_logits, target_probs, weight = target_mask)
        #loss = F.binary_cross_entropy(probs, target, weight = target_mask)
        #loss = F.binary_cross_entropy(background_prob.unsqueeze(1), target, weight = target_mask)

        return loss

class VisibilityLoss(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, prediction, vis_mask):
        return F.binary_cross_entropy_with_logits(prediction, vis_mask)

class CombinedLoss(nn.Module):

    def __init__(self, extrinsics, background_classes, foreground_classes, out_shape = (200, 200), 
            resolution = 0.25, ignore_index = 255, loss_weights=[1, 1, 1], priors=None):
        super().__init__()
        
        self.celoss = CELossWithDistance(extrinsics, out_shape, resolution, ignore_index, priors=priors)
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
        loss_combined = a * celoss + c * vis_loss

        return loss_combined, {"{}/total_loss".format(split): loss_combined.clone().detach().mean(),
                               "{}/seg_loss".format(split): celoss.detach().mean(),
                               "{}/background_loss".format(split): backgroundloss.detach().mean(),
                               "{}/visibility_loss".format(split): vis_loss.detach().mean()}
