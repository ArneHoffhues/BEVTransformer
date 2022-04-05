import albumentations
from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset
from bev.data.nuscenes_utils import NUSCENES_TRAIN_SCENES, NUSCENES_VAL_SCENES, NUSCENES_CAMERA_NAMES, iterate_nuscenes_samples
from nuscenes import NuScenes
import os
import random

class NuscenesSegmentationBase(Dataset):
    def __init__(self, nuscenes_path, label_path, split, dataset_size = None, nuscenes_version='v1.0-trainval', 
            preprocess_param = None, mapping_param = None, n_labels = 7):
        super().__init__()

        self.n_labels = n_labels
        self.nuscenes = NuScenes(nuscenes_version, nuscenes_path)
        
        assert split in ["train", "val"]
        if split == "train":
            scene_names = NUSCENES_TRAIN_SCENES
        else:
            scene_names = NUSCENES_VAL_SCENES
        self.label_path = label_path
        self.labels = self.get_tokens(scene_names)

        if dataset_size is None:
            self.dataset_size = len(self.labels)
        else:
            assert dataset_size <= len(self.labels)
            self.dataset_size = dataset_size
        self.shuffle_samples()

        self.crop_region = [150, 0, 650, 500] if preprocess_param is None else preprocess_param['crop_region']
        self.flip_prob = 0.5 if preprocess_param is None else preprocess_param['flip_prob']
        self.rgb_size = (400, 800) if preprocess_param is None else preprocess_param['rgb_size']
        self.bev_size = (25, 25) if preprocess_param is None else preprocess_param['bev_size']

        #id mapping
        self.unique_ids = [0, 2, 4, 5, 6, 9] if mapping_param is None else mapping_param['unique_ids']
        assert len(self.unique_ids) + 1 == self.n_labels
        self.id_map = np.nan * np.ones((15))
        self.id_map[self.unique_ids] = np.arange(len(self.unique_ids)) + 1
        if mapping_param is not None:
            for k,v in mapping_param['remaps'].items():
                assert v in range(self.n_labels)
                self.id_map[k] = v

        if self.crop_region is not None and len(self.crop_region) == 4:
            x_min, y_min, x_max, y_max = self.crop_region
            self.cropper = albumentations.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
        else:
            self.cropper =albumentations.CenterCrop(height= 800, width = 800)

        self.flip = albumentations.HorizontalFlip(p=self.flip_prob)


        if self.rgb_size is not None and self.bev_size is not None:
            #self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            self.bev_rescaler = albumentations.Resize(*self.bev_size, interpolation=0)
            self.rgb_rescaler = albumentations.Resize(*self.rgb_size)
            self.vflip = albumentations.VerticalFlip(always_apply = True, p=1.0)
            self.bev_preprocessor = albumentations.Compose([self.cropper, self.bev_rescaler, self.vflip], additional_targets={'mask': 'image'})
            self.preprocessor = albumentations.Compose([self.flip], additional_targets={'bev': 'image', 'mask': 'image'})
        else:
            self.preprocessor = lambda **kwargs: kwargs
    
    
    def get_tokens(self, scene_names=None):
        
        self.tokens = list()

        # Iterate over scenes
        for scene in self.nuscenes.scene:
            
            # Ignore scenes which don't belong to the current split
            if scene_names is not None and scene['name'] not in scene_names:
                continue
             
            # Iterate over samples
            for sample in iterate_nuscenes_samples(self.nuscenes, 
                                          scene['first_sample_token']):
                
                # Iterate over cameras
                for camera in NUSCENES_CAMERA_NAMES:
                    self.tokens.append(sample['data'][camera])
        
        return self.tokens


    def shuffle_samples(self):
        self.samples = random.sample(self.labels, self.dataset_size) 

    def __len__(self):
        return len(self.samples)

    def decode_binary_labels(self, labels, n_class=15):
        bits = torch.pow(2, torch.arange(n_class))
        return (labels & bits.view(-1,1,1)) > 0

    def preprocess_image(self, sample):
        token = sample
        img_name = token + ".png"
        labels = to_tensor(Image.open(os.path.join(self.label_path, img_name))).long()
        labels = self.decode_binary_labels(labels)
        bev, mask = labels[:-1], ~labels[-1].numpy().astype(np.uint8)
    
        idx = torch.LongTensor(self.unique_ids)
        bev = bev.index_select(0, idx).numpy()

        bev_composed = np.zeros(bev.shape[-2:], dtype=np.uint8)
        for i in range(bev.shape[0]):
            bev_composed[bev[i, :, :]] = i + 1
        
        bev_preprocessed = self.bev_preprocessor(image = bev_composed, mask = mask)
        bev = bev_preprocessed["image"]
        mask = bev_preprocessed["mask"]

        rgb_img = np.asarray(Image.open(self.nuscenes.get_sample_data_path(token))).astype(np.uint8)
        rgb_img = self.rgb_rescaler(image=rgb_img)["image"]
        rgb_img = (rgb_img/127.5 - 1.0).astype(np.float32)
        preprocessed = self.preprocessor(image = rgb_img, bev = bev, mask =mask)

        # Load camera intrinsics matrix
        sample_data = self.nuscenes.get('sample_data', token)
        sensor = self.nuscenes.get(
            'calibrated_sensor', sample_data['calibrated_sensor_token'])
        intrinsics = torch.tensor(sensor['camera_intrinsic'])

        # Scale calibration matrix to account for image downsampling
        intrinsics[0] *= self.rgb_size[1] / sample_data['width']
        intrinsics[1] *= self.rgb_size[0] / sample_data['height']
        
        data = dict()
        data["image"] =preprocessed["image"]
        data["mask"] = preprocessed["mask"]
        data["cam"] = intrinsics.numpy()

        # one-hot encoding
        image = preprocessed["bev"]
        flatseg = np.ravel(image).astype(int)
        onehot = np.zeros((flatseg.size, self.n_labels), dtype=np.bool)
        onehot[np.arange(flatseg.size), flatseg] = True
        onehot = onehot.reshape(image.shape + (self.n_labels,)).astype(int)
        image = onehot
        data["bev"] = image

        return data

    def __getitem__(self, i):
        example = dict()
        data = self.preprocess_image(self.samples[i])
        example["image"] = data["image"]
        example["mask"] = data["mask"]
        example["bev"] = data["bev"]
        example["cam"] = data["cam"]
        #example["sample"] = self.samples[i]
        return example

class NuscenesSegmentationTrain(NuscenesSegmentationBase):
    def __init__(self, nuscenes_path, label_path, dataset_size = None, nuscenes_version='v1.0-trainval',
            preprocess_param = None, mapping_param = None, n_labels = 7):
        super().__init__(nuscenes_path=nuscenes_path, label_path=label_path, dataset_size=dataset_size,
                nuscenes_version=nuscenes_version, preprocess_param=preprocess_param, mapping_param=mapping_param,
                n_labels=n_labels, split = 'train')

    def get_split(self):
        return "train"

class NuscenesSegmentationValidation(NuscenesSegmentationBase):
    def __init__(self, nuscenes_path, label_path, dataset_size = None, nuscenes_version='v1.0-trainval',
            preprocess_param = None, mapping_param = None, n_labels = 7):
        super().__init__(nuscenes_path=nuscenes_path, label_path=label_path, dataset_size=dataset_size,
                nuscenes_version=nuscenes_version, preprocess_param=preprocess_param, mapping_param=mapping_param,
                n_labels=n_labels, split = 'val')

    def get_split(self):
        return "validation"

