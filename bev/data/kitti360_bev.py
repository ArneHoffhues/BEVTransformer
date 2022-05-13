import numpy as np
import os
import albumentations
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
import json
import umsgpack
from bev.data.kitti360_utils import CropAugmentation, RotateAugmentation

class KITTI360BEVBase(Dataset):
    
    _IMG_DIR = "img"
    _BEV_MSK_DIR = "bev_msk"
    _WEIGHTS_MSK_DIR = "class_weights"
    _BEV_DIR = "bev_ortho"
    _LST_DIR = "split"
    _METADATA_FILE = "metadata_ortho.bin"
    _DEPTH_SCALE_FACTOR = 256

    def __init__(self, rgb_root, bev_root, depth_root = None, with_depth = False, max_depth = 80., 
            n_labels = 10, preprocess_param = None, mapping_param = None, split = None):
        self.split = split
        self.rgb_root = rgb_root
        self.bev_root = bev_root
        self.n_labels = n_labels
        self.rgb_cameras = ['front']
        self.with_depth = with_depth
        self.crop_augmentation = CropAugmentation(rgb_root)
        self.rotate_augmentation = RotateAugmentation()

        if with_depth:
            assert depth_root is not None
            self.depth_root = depth_root
            self.max_depth = max_depth
        
        self.img_dir = os.path.join(bev_root, KITTI360BEVBase._IMG_DIR)
        self.bev_msk_dir = os.path.join(bev_root, KITTI360BEVBase._BEV_MSK_DIR, KITTI360BEVBase._BEV_DIR)
        self.weights_msk_dir = os.path.join(bev_root, KITTI360BEVBase._WEIGHTS_MSK_DIR)
        self.lst_dir = os.path.join(bev_root, KITTI360BEVBase._LST_DIR)

        self.crop_region = [46, 14, 722, 690] if preprocess_param is None or 'crop_region' not in preprocess_param \
                else preprocess_param['crop_region']
        self.flip_prob = 0.5 if preprocess_param is None or 'flip_prob' not in preprocess_param \
                else preprocess_param['flip_prob']
        self.additional_augmentation_prob = 0. if preprocess_param is None or 'additional_augmentation_prob' not in preprocess_param \
                else preprocess_param['additional_augmentation_prob']
        self.do_rotate_augmentation = False if preprocess_param is None or 'do_rotate_augmentation' not in preprocess_param \
                else preprocess_param['do_rotate_augmentation']
        self.do_side_crop_augmentation = False if preprocess_param is None or 'do_side_crop_augmentation' not in preprocess_param \
                else preprocess_param['do_side_crop_augmentation']
        self.do_bottom_crop_augmentation = False if preprocess_param is None or 'do_bottom_crop_augmentation' not in preprocess_param \
                else preprocess_param['do_bottom_crop_augmentation']
        #self.bottom_crop_augmentation_prob = 0.5 if preprocess_param is None or 'bottom_crop_augmentation_prob' not in preprocess_param \
        #        else preprocess_param['bottom_crop_augmentation_prob']
        #self.side_crop_augmentation_prob = 0.5 if preprocess_param is None or 'side_crop_augmentation_prob' not in preprocess_param \
        #        else preprocess_param['side_crop_augmentation_prob']
        #self.rotate_augmentation_prob = 0.5 if preprocess_param is None or 'rotate_augmentation_prob' not in preprocess_param \
        #        else preprocess_param['rotate_augmentation_prob']
        self.rgb_size = (384, 1408) if preprocess_param is None or 'rgb_size' not in preprocess_param \
                else preprocess_param['rgb_size']
        self.bev_size = (200, 200) if preprocess_param is None or 'bev_size' not in preprocess_param \
                else preprocess_param['bev_size']
        self.brightness_factor = 0.2 if preprocess_param is None or 'brightness_factor' not in preprocess_param \
                else preprocess_param['brightness_factor']
        self.contrast_factor = 0.2 if preprocess_param is None or 'contrast_factor' not in preprocess_param \
                else preprocess_param['contrast_factor']
        self.saturation_factor = 0 if preprocess_param is None or 'saturation_factor' not in preprocess_param \
                else preprocess_param['saturation_factor']
        self.hue_factor = 0 if preprocess_param is None or 'hue_factor' not in preprocess_param \
                else preprocess_param['hue_factor']

        #id mapping

        self.unique_ids = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10] \
                if mapping_param is None else mapping_param['unique_ids']
        assert len(self.unique_ids) == self.n_labels
        self.id_map = np.ones((256)) * 255
        self.id_map[self.unique_ids] = np.arange(len(self.unique_ids))
        if mapping_param is not None:
            for k,v in mapping_param['remaps'].items():
                k = int(k.replace("_", ""))
                assert int(v) in range(self.n_labels)
                self.id_map[k] = int(v)
        
        assert split in ['train', 'val']
        self.meta, self.images, self.img_map = self._load_split(split) 
        
        if split == 'train':
            self.shuffled_indices = torch.randperm(len(self.images))
        else:
            self.shuffled_indices = torch.arange(len(self.images))

        if self.crop_region is not None and len(self.crop_region) == 4:
            x_min, y_min, x_max, y_max = self.crop_region
            self.cropper = albumentations.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
        else:
            self.cropper=albumentations.CenterCrop(height= 704, width = 768)

        self.colorjitter = albumentations.ColorJitter(brightness = self.brightness_factor, contrast = self.contrast_factor,
                saturation = self.saturation_factor, hue = self.hue_factor, always_apply = True)
        self.flip = albumentations.HorizontalFlip(p = self.flip_prob)

        assert self.rgb_size is not None and self.bev_size is not None
        #self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
        self.bev_rescaler = albumentations.Resize(*self.bev_size, interpolation=0)
        self.rgb_rescaler = albumentations.Resize(*self.rgb_size)
        self.bev_preprocessor = albumentations.Compose([self.cropper, self.bev_rescaler])
        if self.with_depth:
            self.depth_rescaler = albumentations.Resize(*self.rgb_size, interpolation = 0)
            self.preprocessor = albumentations.Compose([self.flip], additional_targets={'bev': 'image', 'mask': 'image', 'depth': 'image'},
                    keypoint_params=albumentations.KeypointParams(format='xy'))
        else:
            self.preprocessor = albumentations.Compose([self.flip], additional_targets={'bev': 'image', 'mask': 'image'}, 
                    keypoint_params=albumentations.KeypointParams(format='xy'))

    def _load_split(self, split):
        with open(os.path.join(self.bev_root, KITTI360BEVBase._METADATA_FILE), "rb") as fid:
            metadata = umsgpack.unpack(fid, encoding="utf-8")

        with open(os.path.join(self.lst_dir, split + ".txt"), "r") as fid:
            lst = fid.readlines()
            lst = [line.strip() for line in lst]

        img_map = {}
        for camera in self.rgb_cameras:
            with open(os.path.join(self.img_dir, "{}.json".format(camera))) as fp:
                map_list = json.load(fp)
                map_dict = {k: v for d in map_list for k, v in d.items()}
                img_map[camera] = map_dict

        meta = metadata["meta"]
        images = [img_desc for img_desc in metadata["images"] if img_desc["id"] in lst]

        return meta, images, img_map
    
    def shuffle_samples(self):
        self.shuffled_indices = torch.randperm(len(self.images))

    def __len__(self):
        return len(self.images)

    def preprocess_sample(self, img_desc):
        bev_path = os.path.join(self.bev_msk_dir, "{}.png".format(img_desc['id']))
        bev_img = Image.open(bev_path)
        rotated = bev_img.rotate(90, expand = True)
        bev_img = np.asarray(rotated).astype(np.uint8)
        mapping = np.zeros(len(img_desc['cat'])).astype(np.uint8)
        for i, cat in enumerate(img_desc['cat']):
            mapping[i] = cat
        bev_img = mapping[bev_img].astype(np.uint8)
        bev_img = self.id_map[bev_img]
        
        mask_path = os.path.join(self.weights_msk_dir, "{}.png".format(img_desc['id']))
        weights_msk = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(float)
        if weights_msk is not None:
            weights_msk_combined = ((weights_msk[:, :, 0] + (weights_msk[:, :, 1] / 10000)) * 10000).astype(np.int32)
        weight_mask = cv2.rotate(weights_msk_combined, cv2.ROTATE_90_COUNTERCLOCKWISE)    

        rgb_path = os.path.join(self.rgb_root, self.img_map['front']["{}.png".format(img_desc['id'])])
        rgb_img = np.array(Image.open(rgb_path)).astype(np.uint8)
        rgb_img = self.colorjitter(image = rgb_img)["image"]
        H, W = rgb_img.shape[:2]
        
        if self.with_depth:
            depth_path = os.path.join(self.depth_root, "{}_depth.png".format(img_desc['id']))
            depth_img = np.array(Image.open(depth_path)).astype(np.uint16)
            depth_img = depth_img / KITTI360BEVBase._DEPTH_SCALE_FACTOR
        else:
            depth_img = None

        cam = np.array(img_desc['cam_intrinsic']).astype(np.float)
        
        do_additional_augmentation = np.random.rand() < self.additional_augmentation_prob
        if do_additional_augmentation:
            if self.do_rotate_augmentation and not (self.do_side_crop_augmentation or self.do_bottom_crop_augmentation):
                do_rotate = True
                do_crop = False
            elif (self.do_side_crop_augmentation or self.do_bottom_crop_augmentation) and not self.do_rotate_augmentation:
                do_crop = True
                do_rotate = False
            else:
                do_rotate = np.random.rand() < 0.5
                do_crop = not do_rotate
            
            if do_crop:
                if self.do_side_crop_augmentation and not self.do_bottom_crop_augmentation:
                    do_side_crop = True
                    do_bottom_crop = False
                elif not self.do_side_crop_augmentation and self.do_bottom_crop_augmentation:
                    do_side_crop = False
                    do_bottom_crop = True
                else:
                    rnd_val = np.random.rand()
                    if rnd_val < 0.33:
                        do_side_crop = True
                        do_bottom_crop = True
                    elif 0.33 <= rnd_val < 0.66:
                        do_bottom_crop = True
                        do_side_crop = False
                    else:
                        do_bottom_crop = False
                        do_side_crop = True
            else:
                do_side_crop = False
                do_bottom_crop = False
        else:
            do_rotate = False
            do_side_crop = False
            do_bottom_crop = False

        #do_rotate = np.random.rand() < self.rotate_augmentation_prob
        if do_rotate:
            rgb_img, bev_img, depth_img, weight_mask, cam = self.rotate_augmentation.do_rotate(rgb_img, bev_img, depth_img, weight_mask, cam)
        
        #do_side_crop = np.random.rand() < self.side_crop_augmentation_prob
        #do_bottom_crop = np.random. rand() < self.bottom_crop_augmentation_prob
        if do_side_crop or do_bottom_crop:
            rgb_img, bev_img, depth_img, weight_mask, cam = self.crop_augmentation.do_crop(do_bottom_crop, do_side_crop,
                    rgb_img, bev_img, depth_img, weight_mask, cam, img_desc['id'])
        
        bev_img = self.bev_preprocessor(image = bev_img)["image"]
        mask = self.bev_preprocessor(image = weight_mask)["image"] 
        rgb_img = self.rgb_rescaler(image=rgb_img)["image"]
        rgb_img = (rgb_img/127.5 - 1.0).astype(np.float32)
        scaling_y, scaling_x = self.rgb_size[0] / H, self.rgb_size[1] / W
        cam = np.array([cam[0] * scaling_x, cam[1] * scaling_y, cam[2]]).astype(float)
        principal_point = [(cam[0,2], cam[1, 2])]

        if self.with_depth:
            depth_img = depth_img / self.max_depth
            depth_img = self.depth_rescaler(image=depth_img)["image"]
            preprocessed = self.preprocessor(image = rgb_img, bev = bev_img, mask =mask, depth = depth_img, keypoints=principal_point)
        else:
            preprocessed = self.preprocessor(image = rgb_img, bev = bev_img, mask =mask, keypoints=principal_point)
        cx_flipped, cy_flipped = preprocessed['keypoints'][0]
        cam[0,2] = cx_flipped
        cam[1,2] = cy_flipped

        data = dict()
        data["image"] = preprocessed["image"]
        data["mask"] = preprocessed["mask"]
        data['cam'] = cam
        data['bev'] = preprocessed["bev"]

        if self.with_depth:
            data['depth'] = preprocessed['depth']
        
        return data

    def __getitem__(self, i):
        example = dict()
        preprocessed = self.preprocess_sample(self.images[self.shuffled_indices[i]])
        example["image"] = preprocessed["image"]
        example["mask"] = preprocessed["mask"]
        example["bev"] = preprocessed["bev"]
        example["cam"] = preprocessed["cam"]
        if self.with_depth:
            example['depth'] = preprocessed['depth']
        return example

class KITTI360BEVTrain(KITTI360BEVBase):
    def __init__(self, rgb_root, bev_root, depth_root = None, with_depth = False, max_depth = 80., n_labels = 10, preprocess_param = None, mapping_param = None):
        super().__init__(rgb_root=rgb_root, bev_root = bev_root, depth_root = depth_root, with_depth = with_depth, max_depth = max_depth,
                n_labels = n_labels, split = 'train', preprocess_param=preprocess_param, mapping_param=mapping_param)

    def get_split(self):
        return "train"

class KITTI360BEVValidation(KITTI360BEVBase):
    def __init__(self, rgb_root, bev_root, depth_root = None, with_depth = False, max_depth = 80.,n_labels = 10, preprocess_param = None, mapping_param = None):
        super().__init__(rgb_root=rgb_root, bev_root = bev_root, depth_root = depth_root, with_depth = with_depth, max_depth = max_depth,
                n_labels = n_labels, split = 'val', preprocess_param=preprocess_param, mapping_param=mapping_param)

    def get_split(self):
        return "validation"

