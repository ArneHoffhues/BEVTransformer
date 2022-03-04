import numpy as np
import os
import albumentations
from PIL import Image
from torch.utils.data import Dataset
import cv2

class KITTIBEVBase(Dataset):
    def __init__(self, rgb_root="", bev_root="",  n_labels = 11, preprocess_param = None, mapping_param = None, split = None):
        self.split = split
        self.rgb_root = rgb_root
        self.bev_root = bev_root
        self.n_labels = n_labels

        self.crop_region = [300, 600, 1300, 1580] if preprocess_param is None else preprocess_param['crop_region']
        self.flip_prob = 0.5 if preprocess_param is None else preprocess_param['flip_prob']
        self.rgb_size = (256, 1024) if preprocess_param is None else preprocess_param['rgb_size']
        self.bev_size = (49, 50) if preprocess_param is None else preprocess_param['bev_size']

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

        if self.rgb_size is not None and self.bev_size is not None:
            #self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            self.bev_rescaler = albumentations.Resize(*self.bev_size, interpolation=0)
            self.rgb_rescaler = albumentations.Resize(*self.rgb_size)
            self.preprocessor = albumentations.Compose([self.flip], additional_targets={'bev': 'image', 'mask': 'image'})
        else:
            self.preprocessor = lambda **kwargs: kwargs

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

        preprocessed = self.preprocessor(image = rgb_img, bev = bev_img, mask =mask)

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

        return data

    def __getitem__(self, i):
        example = dict()
        preprocessed = self.preprocess_sample(self.samples[i])
        example["image"] = preprocessed["image"]
        example["mask"] = preprocessed["mask"]
        example["bev"] = preprocessed["bev"]
        example["cam"] = preprocessed["cam"]
        example["sample"] = self.samples[i]
        return example

class KITTIBEVTrain(KITTIBEVBase):
    def __init__(self, rgb_root, bev_root, n_labels, split, preprocess_param = None, mapping_param = None):
        super().__init__(rgb_root=rgb_root, bev_root = bev_root, n_labels = n_labels, split = split, 
                preprocess_param=preprocess_param, mapping_param=mapping_param)

    def get_split(self):
        return "train"

class KITTIBEVValidation(KITTIBEVBase):
    def __init__(self, rgb_root, bev_root, n_labels, split, preprocess_param = None, mapping_param = None):
        super().__init__(rgb_root=rgb_root, bev_root = bev_root, n_labels = n_labels, split = split, 
                preprocess_param=preprocess_param, mapping_param=mapping_param)

    def get_split(self):
        return "validation"

