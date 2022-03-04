from torchmetrics import Metric
import torch

class ConfusionMatrixMetric(Metric):

    def __init__(self, num_class, class_names=None, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if class_names is not None:
            assert len(class_names) == num_class, "Number of classes and class names do not match."
        self.class_names = class_names
        self.num_class = num_class

        self.add_state('tp', default=torch.zeros(num_class, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state('fp', default=torch.zeros(num_class, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state('fn', default=torch.zeros(num_class, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state('tn', default=torch.zeros(num_class, dtype=torch.long), dist_reduce_fx="sum")
    
    def update(self, preds, labels, mask=None):
        #preds, target, mask = self._input_format(preds, target, mask)
        assert preds.shape == labels.shape

        preds = preds.detach()
        labels = labels.detach()

        preds = preds.flatten(2, -1).permute(1, 0, 2).reshape(
            self.num_class, -1)
        labels = labels.flatten(2, -1).permute(1, 0, 2).reshape(
            self.num_class, -1)
        
        if mask is not None:
            preds = preds[:, mask.flatten()]
            labels = labels[:, mask.flatten()]
        
        true_pos = preds & labels
        false_pos = preds & ~labels
        false_neg = ~preds & labels
        true_neg = ~preds & ~labels

        # Update global counts
        self.tp += true_pos.long().sum(-1)
        self.fp += false_pos.long().sum(-1)
        self.fn += false_neg.long().sum(-1)
        self.tn += true_neg.long().sum(-1)
    
    def compute(self, split):
        iou_dict = dict()
        ious = self.tp.float() / (self.tp + self.fn + self.fp).float()
        if self.class_names is not None:
            for name, iou_score in zip(self.class_names, ious):
               iou_dict[f'{split}/IoU/{name}'] = iou_score
        valid = (self.tp + self.fn) > 0
        if not valid.any():
            mean = 0.
        else:
            mean = float(ious[valid].mean())
        iou_dict[f'{split}/IoU/MEAN'] = mean
        return iou_dict

