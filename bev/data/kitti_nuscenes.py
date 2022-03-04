from bev.data.nuscenes_bev import NuscenesSegmentationBase
from bev.data.kitti_bev import KITTIBEVBase
import torch
from torch.utils.data import Dataset

class KITTIPlusNuscenesTrain(Dataset):

    def __init__(self, kitti_root, kitti_label_root, kitti_scenes, nuscenes_root, nuscenes_label_root, n_labels = 7):

        kitti_preprocess = dict()
        kitti_preprocess["crop_region"] = [300, 580, 1300, 1580]
        kitti_preprocess["flip_prob"] = 0.5
        kitti_preprocess["rgb_size"] = (400, 800)
        kitti_preprocess["bev_size"] = (25, 25)
        kitti_mapping = dict()
        kitti_mapping["unique_ids"] = [0, 40, 48, 10, 18, 13, 30]
        kitti_mapping["remaps"] = {'152': 3, '154': 6, '157': 5, '158': 4}
        self.kitti = KITTIBEVBase(rgb_root = kitti_root,  bev_root = kitti_label_root, n_labels = n_labels, 
                preprocess_param = kitti_preprocess, mapping_param = kitti_mapping, split = kitti_scenes)
        self.nuscenes = NuscenesSegmentationDataset(nuscenes_path = nuscenes_root, label_path = nuscenes_label_root, 
                dataset_size = len(self.kitti), n_labels = n_labels)
        self.concatenated = torch.utils.data.ConcatDataset([self.kitti, self.nuscenes])
    
    def __len__(self):
        return len(self.concatenated)
    
    def __getitem__(self, i):
        return self.concatenated[i]

    def shuffle_samples(self):
       self.nuscenes.shuffle_samples() 

    def get_split(self):
        return "train"

