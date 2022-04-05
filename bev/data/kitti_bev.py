import numpy as np
import os
import albumentations
from PIL import Image
from torch.utils.data import Dataset
import cv2

class KITTIBEVBase(Dataset):

    _DEPTH_SCALE_FACTOR = 256

    def __init__(self, rgb_root="", bev_root="", depth_root=None, n_labels = 11, preprocess_param = None, 
            mapping_param = None, split = None, with_depth=False, max_depth = 80.):
        self.split = split
        self.rgb_root = rgb_root
        self.bev_root = bev_root
        self.n_labels = n_labels
        self.with_depth = with_depth

        if with_depth:
            assert depth_root is not None
            self.depth_root = depth_root
            self.max_depth = max_depth 

        self.crop_region = [300, 600, 1300, 1580] if preprocess_param is None else preprocess_param['crop_region']
        self.flip_prob = 0.5 if preprocess_param is None else preprocess_param['flip_prob']
        self.rgb_size = (256, 1024) if preprocess_param is None else preprocess_param['rgb_size']
        self.bev_size = (50, 50) if preprocess_param is None else preprocess_param['bev_size']

        #id mapping

        self.unique_ids = [0, 10, 13, 18, 30, 40, 48, 51, 72, 70, 50] \
                if mapping_param is None else mapping_param['unique_ids']
        assert len(self.unique_ids) == self.n_labels
        self.id_map = np.zeros((160))
        self.id_map[self.unique_ids] = np.arange(len(self.unique_ids))
        if mapping_param is not None:
            for k,v in mapping_param['remaps'].items():
                k = int(k.replace("_", ""))
                assert int(v) in range(self.n_labels)
                self.id_map[k] = int(v)
        else:
            self.id_map[152] = 1
            self.id_map[154] = 4
            self.id_map[157] = 2
            self.id_map[158] = 3

        self.samples = []
        for seq in self.split:
            dir_path = os.path.join(self.bev_root, seq, 'semantic')
            images = os.listdir(dir_path)
            for img in images:
                self.samples += [(str(seq), str(img))]

        if self.crop_region is not None and len(self.crop_region) == 4:
            x_min, y_min, x_max, y_max = self.crop_region
            self.cropper = albumentations.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
        else:
            self.cropper=albumentations.CenterCrop(height= 1600, width = 1600)

        self.flip = albumentations.HorizontalFlip(p=self.flip_prob)

        assert self.rgb_size is not None and self.bev_size is not None
        #self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
        self.bev_rescaler = albumentations.Resize(*self.bev_size, interpolation=0)
        self.rgb_rescaler = albumentations.Resize(*self.rgb_size)
        if self.with_depth:
            self.depth_rescaler = albumentations.Resize(*self.rgb_size, interpolation = 0)
            self.preprocessor = albumentations.Compose([self.flip], additional_targets={'bev': 'image', 'mask': 'image', 'depth': 'image'},
                    keypoint_params=albumentations.KeypointParams(format='xy'))
        else:
            self.preprocessor = albumentations.Compose([self.flip], additional_targets={'bev': 'image', 'mask': 'image'},
                    keypoint_params=albumentations.KeypointParams(format='xy'))

    def __len__(self):
        return len(self.samples)

    def preprocess_sample(self, sample):
        bev_path = os.path.join(self.bev_root, sample[0], 'semantic',  sample[1])
        bev_img = np.asarray(Image.open(bev_path)).astype(np.uint8)
        bev_img = self.cropper(image = bev_img)["image"]

        mask_path = os.path.join(self.bev_root, sample[0], 'visibility', sample[1].split('_')[-1])
        mask = np.asarray(Image.open(mask_path)).astype(np.uint8)
        mask = self.cropper(image = mask)["image"]        

        bev_img = self.id_map[bev_img]
        #bev_img[~mask] = 0
        bev_img = self.bev_rescaler(image = bev_img)["image"]
        mask = self.bev_rescaler(image = mask)["image"]

        rgb_path = os.path.join(self.rgb_root, sample[0], 'image_2', sample[1].split("_")[-1])
        rgb_img = np.asarray(Image.open(rgb_path)).astype(np.uint8)
        H, W = rgb_img.shape[:2]
        rgb_img = self.rgb_rescaler(image=rgb_img)["image"]
        rgb_img = (rgb_img/127.5 - 1.0).astype(np.float32)

        #scaling the camera matrix
        calib_path = os.path.join(self.rgb_root, sample[0], 'calib.txt')
        with open(calib_path, 'r') as file:
            data = file.readlines()
        arr = []
        for elem in data:
            if elem.startswith('P2'):
                elem = elem.replace('\n', '')
                elem = elem.replace('P2: ', '')
                arr = np.asarray(elem.split(' ')).astype(np.float32)
                break
        arr = arr.reshape(3, 4)
        mat = cv2.decomposeProjectionMatrix(arr)
        cam = mat[0]
        scaling_y, scaling_x = self.rgb_size[0] / H, self.rgb_size[1] / W
        cam = np.array([cam[0] * scaling_x, cam[1] * scaling_y, cam[2]]).astype(float)
        principal_point = [(cam[0,2], cam[1, 2])]

        if self.with_depth:
            depth_path = os.path.join(self.depth_root, sample[0], sample[1])
            depth_img = np.array(Image.open(depth_path)).astype(np.uint16)
            depth_img = depth_img / KITTIBEVBase._DEPTH_SCALE_FACTOR
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

        # one-hot encoding
        image = preprocessed['bev']
        flatseg = np.ravel(image).astype(int)
        onehot = np.zeros((flatseg.size, self.n_labels), dtype=np.bool)
        onehot[np.arange(flatseg.size), flatseg] = True
        onehot = onehot.reshape(image.shape + (self.n_labels,)).astype(int)
        image = onehot
        data['bev'] = image

        if self.with_depth:
            data['depth'] = preprocessed['depth']

        return data

    def __getitem__(self, i):
        example = dict()
        preprocessed = self.preprocess_sample(self.samples[i])
        example["image"] = preprocessed["image"]
        example["mask"] = preprocessed["mask"]
        example["bev"] = preprocessed["bev"]
        example["cam"] = preprocessed["cam"]
        if self.with_depth:
            example['depth'] = preprocessed['depth']
        example["sample"] = self.samples[i]
        return example

class KITTIBEVTrain(KITTIBEVBase):
    def __init__(self, rgb_root, bev_root, n_labels, split, depth_root= None, preprocess_param = None, 
            mapping_param = None, with_depth=False, max_depth = 80.):
        super().__init__(rgb_root=rgb_root, bev_root = bev_root, n_labels = n_labels, split = split, depth_root = depth_root, 
                preprocess_param=preprocess_param, mapping_param=mapping_param, with_depth = with_depth, max_depth = max_depth)

    def get_split(self):
        return "train"

class KITTIBEVValidation(KITTIBEVBase):
    def __init__(self, rgb_root, bev_root, n_labels, split, depth_root=None, preprocess_param = None, 
            mapping_param = None, with_depth=False, max_depth = 80.):
        super().__init__(rgb_root=rgb_root, bev_root = bev_root, n_labels = n_labels, split = split, depth_root = depth_root,
                preprocess_param=preprocess_param, mapping_param=mapping_param, with_depth = with_depth, max_depth = max_depth)

    def get_split(self):
        return "validation"

