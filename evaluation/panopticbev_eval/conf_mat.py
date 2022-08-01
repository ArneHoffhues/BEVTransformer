import torch

def confusion_matrix(num_classes, sem_pred, sem, ignore_index=255):
    confmat = sem.new_zeros(num_classes * num_classes, dtype=torch.float)

    valid = sem != ignore_index
    if valid.any():
        sem_pred = sem_pred[valid]
        sem = sem[valid]

        confmat.index_add_(0, sem.view(-1) * num_classes + sem_pred.view(-1),
                    confmat.new_ones(sem.numel()))

    return confmat.view(num_classes, num_classes)
