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
import cv2

class NuscenesSegmentationBase(Dataset):
    def __init__(self, label_path, split, depth_path = None, dataset_size= None, preprocess_param = None, n_labels = 14):
        super().__init__()

        self.n_labels = n_labels
        self.dataset_size = dataset_size
        self.split = split

        assert split in ["train", "val"]
        if split == "train":
            self.scene_names = NUSCENES_TRAIN_SCENES
        else:
            self.scene_names = NUSCENES_VAL_SCENES
        self.label_path = label_path
        self.depth_path = depth_path

        self.crop_region = [150, 0, 650, 500] if preprocess_param is None else preprocess_param['crop_region']
        self.flip_prob = 0.5 if preprocess_param is None else preprocess_param['flip_prob']
        self.rgb_size = (600, 800) if preprocess_param is None else preprocess_param['rgb_size']
        self.bev_size = (200, 200) if preprocess_param is None else preprocess_param['bev_size']

        if self.crop_region is not None and len(self.crop_region) == 4:
            x_min, y_min, x_max, y_max = self.crop_region
            self.cropper = albumentations.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
        else:
            self.cropper =albumentations.CenterCrop(height= 800, width = 800)

        self.flip = albumentations.HorizontalFlip(p=self.flip_prob)
        self.vflip = albumentations.VerticalFlip(p=1.0)

        if self.rgb_size is not None and self.bev_size is not None:
            #self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            self.bev_rescaler = albumentations.Resize(*self.bev_size, interpolation=0)
            self.rgb_rescaler = albumentations.Resize(*self.rgb_size)
            self.bev_preprocessor = albumentations.Compose([self.cropper, self.bev_rescaler, self.vflip], additional_targets={'mask': 'image'})
            if self.depth_path is not None: 
                self.preprocessor = albumentations.Compose([self.flip], additional_targets={'bev': 'image', 'mask': 'image', 'depth': 'image'})
            else:
                self.preprocessor = albumentations.Compose([self.flip], additional_targets={'bev': 'image', 'mask': 'image'})
        else:
            self.preprocessor = lambda **kwargs: kwargs
    
    def set_nuscenes(self, nuscenes):
        self.nuscenes = nuscenes
        self.labels = self.get_tokens(self.scene_names)

        if self.dataset_size is None:
            self.dataset_size = len(self.labels)
        else:
            assert self.dataset_size <= len(self.labels)

        if self.split == 'train':
            self.shuffle_samples()
        else:
            self.shuffled_indices = torch.arange(len(self.labels))

    
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
        self.shuffled_indices = torch.randperm(len(self.labels))
        if not self.dataset_size == len(self.labels):
            self.shuffled_indices = self.shuffled_indices[:self.dataset_size] 

    def __len__(self):
        return self.dataset_size

    def decode_binary_labels(self, labels, n_class=15):
        bits = torch.pow(2, torch.arange(n_class))
        return (labels & bits.view(-1,1,1)) > 0

    def preprocess_image(self, sample):
        token = sample
        img_name = token + ".png"
        labels = to_tensor(Image.open(os.path.join(self.label_path, img_name))).long()
        labels = self.decode_binary_labels(labels)
        bev, mask = labels[:-1].numpy().astype(np.uint8), (~labels[-1].numpy()).astype(np.uint8)
        bev = np.moveaxis(bev, 0, -1)

        bev_preprocessed = self.bev_preprocessor(image = bev, mask = mask)
        bev = bev_preprocessed["image"]
        mask = bev_preprocessed["mask"]
        #bev = cv2.rotate(bev_preprocessed["image"], cv2.ROTATE_180)
        #mask = cv2.rotate(bev_preprocessed["mask"], cv2.ROTATE_180)

        rgb_img = np.asarray(Image.open(self.nuscenes.get_sample_data_path(token))).astype(np.uint8)
        rgb_img = self.rgb_rescaler(image=rgb_img)["image"]
        rgb_img = (rgb_img/127.5 - 1.0).astype(np.float32)
        
        if self.depth_path is not None:
            depth_img_name = token +'_depth.png'
            depth_img = np.asarray(Image.open(os.path.join(self.depth_path, depth_img_name))).astype(np.uint16)
            depth_img = depth_img / 256.
            depth_img = self.rgb_rescaler(image=depth_img)["image"]
            preprocessed = self.preprocessor(image = rgb_img, bev = bev, mask =mask, depth = depth_img)
        else:
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
        data["bev"] = preprocessed["bev"]
        data["mask"] = preprocessed["mask"]
        if self.depth_path is not None:
            data["depth"] = preprocessed["depth"]
        data["cam"] = intrinsics.numpy()

        # one-hot encoding
        #image = preprocessed["bev"]
        #flatseg = np.ravel(image).astype(int)
        #onehot = np.zeros((flatseg.size, self.n_labels), dtype=np.bool)
        #onehot[np.arange(flatseg.size), flatseg] = True
        #onehot = onehot.reshape(image.shape + (self.n_labels,)).astype(int)
        #image = onehot
        #data["bev"] = image

        return data

    def __getitem__(self, i):
        example = dict()
        data = self.preprocess_image(self.labels[self.shuffled_indices[i]])
        example["image"] = data["image"]
        example["mask"] = data["mask"]
        example["bev"] = data["bev"]
        if self.depth_path is not None:
            example["depth"] = data["depth"]
        example["cam"] = data["cam"]
        #example["sample"] = self.samples[i]
        return example

class NuscenesSegmentationTrain(NuscenesSegmentationBase):
    def __init__(self, label_path, depth_path=None, dataset_size = None, preprocess_param = None, n_labels = 14):
        super().__init__(label_path=label_path, depth_path = depth_path, dataset_size=dataset_size, preprocess_param=preprocess_param, 
                n_labels=n_labels, split = 'train')

    def get_split(self):
        return "train"

class NuscenesSegmentationValidation(NuscenesSegmentationBase):
    def __init__(self, label_path, depth_path=None, dataset_size = None, preprocess_param = None, n_labels = 14):
        super().__init__(label_path=label_path, depth_path =depth_path, dataset_size=dataset_size, preprocess_param=preprocess_param, 
                n_labels=n_labels, split = 'val')

    def get_split(self):
        return "validation"

