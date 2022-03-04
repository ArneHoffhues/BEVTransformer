import math
import logging
import torch
from torch import optim

def restore_snapshot(net, snapshot, ignore_keys=[]):
    """
    Restore weights and optimizer (if needed ) for resuming job.
    """
    sd = torch.load(snapshot, map_location=torch.device('cpu'))["state_dict"]
    print("SFSegNet checkpoint loaded from {}.".format(snapshot))
    
    sd_copy = sd.copy()

    for k in sd.keys():
        for ik in ignore_keys:
            if k.startswith(ik):
                print("Deleting key {} from state_dict".format(k))
                del sd_copy[k]
    net.load_state_dict(sd_copy, strict=False)
    return net


