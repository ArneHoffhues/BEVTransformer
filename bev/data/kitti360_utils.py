import os
from PIL import Image
import numpy as np
import umsgpack
import albumentations
import cv2

class CropAugmentation():
    
    _EPSILON = -1e-6

    def __init__(self, kitti_path, cropping_range=[0.2, 0.3], ignore_index=255, pixel_resolution=0.074):
        
        self.kitti_path = kitti_path
        self.cropping_range = cropping_range
        self.ignore_index = ignore_index
        self.pixel_resolution = pixel_resolution

        intrinsic_file = os.path.join(kitti_path, 'calibration', 'perspective.txt')
        #fileCameraToVelo = os.path.join(kitti_path, 'calibration', 'calib_cam_to_velo.txt')
        #TODO change file back to original relative to dataset path
        fileCameraToVelo = os.path.abspath('/home/hoffhues/KITTI-360_calib/calibration/calib_cam_to_velo.txt')

        intrinsic_loaded = False
        with open(intrinsic_file) as f:
            intrinsics = f.read().splitlines()
        for line in intrinsics:
            line = line.split(' ')
            if line[0] == 'P_rect_00:':
                K = [float(x) for x in line[1:]]
                self.K = np.reshape(K, [3,4])
                intrinsic_loaded = True
            elif line[0] == 'R_rect_00:':
                R_rect = np.eye(4)
                R_rect[:3,:3] = np.array([float(x) for x in line[1:]]).reshape(3,3)
            elif line[0] == "S_rect_00:":
                self.width = int(float(line[1]))
                self.height = int(float(line[2]))

        assert(intrinsic_loaded==True)
        assert(self.width>0 and self.height>0)
        
        camToVelo = np.concatenate((np.loadtxt(fileCameraToVelo).reshape(3,4), np.array([0, 0, 0, 1]).reshape(1,4)))
        veloToCam = np.linalg.inv(camToVelo)
        self.veloToRect = np.matmul(R_rect, veloToCam)

        self.resize = albumentations.Resize(self.height, self.width)
    
    def _loadDepthFromLaser(self, sample):
        seq, img = sample.split(";")
        scan_file = os.path.join(self.kitti_path, 'data_3d_raw', seq, 'velodyne_points', 'data', img + '.bin')

        pcd = np.fromfile(scan_file, dtype = np.float32)
        pcd = np.reshape(pcd, [-1, 4])
        pcd[:, 3] = 1

        points_cam = np.matmul(self.veloToRect, pcd.T).T
        points_cam = points_cam[:, :3]
        points_proj = np.matmul(self.K[:3, :3].reshape([1, 3, 3]), points_cam.T)

        depth = points_proj[:,2,:]
        depth[depth==0] = -1e-6
        u = np.round(points_proj[:,0,:]/np.abs(depth)).astype(np.uint16)
        v = np.round(points_proj[:,1,:]/np.abs(depth)).astype(np.uint16)

        depth_map = np.zeros((self.height, self.width), dtype = np.float32)
        mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<self.width), v>=0), v<self.height)
        mask = np.logical_and(np.logical_and(mask, depth>0), depth<80)
        depth_map[v[mask], u[mask]] = depth[mask]
        
        return depth_map

    def _do_bottom_crop(self, rgb, bev, depth_img, depth_map, mask, intrinsics):
        crop_factor = np.random.uniform(low = self.cropping_range[0], high = self.cropping_range[1])
        
        proj_point = (int(self.height - crop_factor * self.height), self.width // 2)

        search_area = depth_map[proj_point[0]-2:proj_point[0]+3, proj_point[1]-2:proj_point[1]+3]
        if np.count_nonzero(search_area) > 0:
            non_zeros = np.nonzero(search_area)
            first_y, first_x = non_zeros[0][0], non_zeros[1][0]
            crop_depth = depth_map[proj_point[0] -2 + first_y, proj_point[1] - 2 + first_x]
        else:
            return rgb, bev, depth_img, mask, intrinsics

        crop = albumentations.Crop(x_min=0, y_min= 0, x_max=self.width, y_max= int(self.height - crop_factor * self.height))
        if depth_img is not None:
            rgb_pipeline = albumentations.Compose([crop, self.resize], additional_targets={'depth': 'image'})
            transformed = rgb_pipeline(image=rgb, depth= depth_img)
            rgb_resized = transformed["image"]
            depth_shifted = transformed["depth"] - crop_depth
        else:
            rgb_cropped = crop(image=rgb)["image"]
            rgb_resized = self.resize(image=rgb_cropped)["image"]
            depth_shifted = None

        bev_crop = int(crop_depth / self.pixel_resolution)
        bev = bev[:bev.shape[0]-bev_crop, :]
        bev_extension = np.ones((bev_crop, bev.shape[1]), dtype = np.uint8) * self.ignore_index
        bev = np.concatenate((bev_extension, bev), axis=0)
        
        mask = mask[:mask.shape[0]-bev_crop, :]
        mask_extension = np.zeros((bev_crop, mask.shape[1]), dtype = np.int32)
        mask = np.concatenate((mask_extension, mask), axis=0)

        intrinsics[1, :] = intrinsics[1, :] * (1 / (1 - crop_factor))
        
        return rgb_resized, bev, depth_shifted, mask, intrinsics
        
    def _do_side_crop(self, rgb, bev, depth_img, depth_map, intrinsics):
        crop_factor = np.random.uniform(low = self.cropping_range[0], high = self.cropping_range[1])
        do_left_side = np.random.rand() < 0.5

        if do_left_side:
            proj_point = (self.height // 2, int(crop_factor * self.width))
        else:
            proj_point = (self.height // 2, int(self.width - crop_factor * self.width))
        
        search_area = depth_map[proj_point[0]-2:proj_point[0]+3, proj_point[1]-2:proj_point[1]+3]
        if np.count_nonzero(search_area) > 0:
            non_zeros = np.nonzero(search_area)
            first_y, first_x = non_zeros[0][0], non_zeros[1][0]
            crop_depth = depth_map[proj_point[0] -2 + first_y, proj_point[1] - 2 + first_x]
        else:
            return rgb, bev, depth_img, intrinsics
        
        if do_left_side:
            crop = albumentations.Crop(x_min=int(crop_factor * self.width), y_min= 0, x_max=self.width, y_max= self.height)
        else:
            crop = albumentations.Crop(x_min=0, y_min= 0, x_max=int(self.width - crop_factor * self.width), y_max= self.height)

        if depth_img is not None:
            rgb_pipeline = albumentations.Compose([crop, self.resize], additional_targets={'depth': 'image'})
            transformed = rgb_pipeline(image=rgb, depth= depth_img)
            rgb_resized = transformed["image"]
            depth_resized = transformed["depth"]
        else:
            rgb_cropped = crop(image=rgb)["image"]
            rgb_resized = self.resize(image=rgb_cropped)["image"]
            depth_resized = None
        
        x_space = ((proj_point[1] - self.K[0, 2]) / self.K[0, 0]) * crop_depth
        z_space = crop_depth
        
        z_dim = np.arange(bev.shape[0])
        z_dim = np.expand_dims(z_dim, axis=1)
        mask = np.ones_like(bev) * (bev.shape[0] - 1) - z_dim
        if do_left_side:
            mask[:, (bev.shape[1] // 2):] = -1
        else:
            mask[:, :(bev.shape[1] // 2 + 1)] = -1
        mid_distance = np.abs(np.arange(bev.shape[1]) - bev.shape[1] // 2)
        mid_distance = np.expand_dims(mid_distance, axis = 0)
        mask = mask / (mid_distance + CropAugmentation._EPSILON)
        mask_bool = np.logical_and(mask <= (z_space / x_space), mask >= 0)
        bev[mask_bool] = self.ignore_index

        if do_left_side:
            intrinsics[0, 2] = intrinsics[0, 2] - (crop_factor * self.width)
        intrinsics[0, :] = intrinsics[0, :] * (1 / (1- crop_factor))

        return rgb_resized, bev, depth_resized, intrinsics

    def do_crop(self, do_bottom_crop, do_side_crop, rgb, bev, depth_img, mask, intrinsics, sample):
        
        depth_map = self._loadDepthFromLaser(sample)

        if do_side_crop:
            rgb, bev, depth_img, intrinsics = self._do_side_crop(rgb, bev, depth_img, depth_map, intrinsics)

        if do_bottom_crop:
            rgb, bev, depth_img, mask, intrinsics = self._do_bottom_crop(rgb, bev, depth_img, depth_map, mask, intrinsics)

        return rgb, bev, depth_img, mask, intrinsics

class RotateAugmentation():

    _EPSILON = -1e-6

    def __init__(self, rotation_range=[-15, 15], ignore_index=255):

        self.rotation_range = rotation_range
        self.ignore_index = ignore_index

    def do_rotate(self, rgb, bev, depth_img, weight_mask, intrinsics):

        angle = np.random.uniform(low=self.rotation_range[0], high = self.rotation_range[1])
        rad = np.radians(angle)
        rgb_size = rgb.shape
        resize = albumentations.Resize(rgb_size[0], rgb_size[1])

        proj_depth = np.ones((rgb_size[0], rgb_size[1]), dtype = np.float32)

        fx, fy, cx, cy = intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]
        rows, cols = proj_depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)

        x = (c - cx) / fx
        y = (r - cy) / fy
        z = proj_depth
        x = x * z
        y = y * z
        coord = np.dstack((x,y, z))
        if angle <= 0:
            crop_coord = np.append(coord[rows // 2, cols -1, :], [1.])
        else:
            crop_coord = np.append(coord[rows // 2, 0, :], [1.])
        coord = np.concatenate((coord, np.ones((rows, cols, 1))), axis = -1).reshape(-1, 4)
        
        rot = np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]])
        trans = np.array([0., 0., 0.]).reshape(-1, 1)
        TrMat = np.concatenate((np.concatenate((rot, trans), axis = 1), np.array([0., 0., 0., 1.]).reshape(1, -1)))

        points_new = np.matmul(TrMat, coord.T).T
        points_new = points_new[:,:3]
        points_proj = np.matmul(intrinsics.reshape([1, 3, 3]), points_new.T)

        crop_proj = np.matmul(intrinsics.reshape([1, 3, 3]), np.matmul(TrMat, crop_coord.T).T[:3].T)
        crop_x = (crop_proj[0, 0] / np.abs(crop_proj[0, 2] + 1e-6)).astype(np.uint16)
        
        depth = points_proj[:,2, :]
        depth[depth==0] = -1e-6
        u = np.round(points_proj[:,0, :]/np.abs(depth)).astype(np.uint16).squeeze(0).reshape(rows, cols)
        v = np.round(points_proj[:,1, :]/np.abs(depth)).astype(np.uint16).squeeze(0).reshape(rows, cols)

        new_img = np.zeros((rgb_size[0], rgb_size[1], 3), dtype = np.uint8)
        mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<cols), v>=0), v<rows)
        new_img[v[mask], u[mask], :] = rgb[mask,:]

        intpol_mask = np.logical_and(np.logical_and(new_img[:, :, 0] == 0, new_img[:, :, 1] == 0), new_img[:,:,2] == 0)
        if angle <= 0:
            intpol_mask[:, cols // 2:] = False
        else:
            intpol_mask[:, :(cols // 2 + 1)] = False
        masked = np.where(intpol_mask == True)
        up = masked[0] - 1
        down = masked[0] + 1
        left = masked[1] - 1
        right = masked[1] + 1
        up[up < 0] = 0
        left[left < 0] = 0
        down[down >= rgb_size[0]] = rgb_size[0] - 1
        right[right >= rgb_size[1]] = rgb_size[1] - 1
        count_masked_neighbours = intpol_mask[up, left].astype(np.uint8) + intpol_mask[up, right].astype(np.uint8) \
                    + intpol_mask[down, left].astype(np.uint8) + intpol_mask[down, right].astype(np.uint8)
        count_unmasked_neighbours = 4 - count_masked_neighbours

        only_masked_neighbours = count_unmasked_neighbours == 0
        intpol_mask[masked[0][only_masked_neighbours], masked[1][only_masked_neighbours]] = False
        del_ind = np.where(only_masked_neighbours)
        up = np.delete(up, del_ind)
        down = np.delete(down, del_ind)
        left = np.delete(left, del_ind)
        right = np.delete(right, del_ind)
        count_unmasked_neighbours = np.delete(count_unmasked_neighbours, del_ind).astype(np.float32)

        new_img[intpol_mask] = ((new_img[up, left].astype(np.float32) + new_img[up, right].astype(np.float32) + new_img[down, left].astype(np.float32) \
                       + new_img[down, right].astype(np.float32)) / count_unmasked_neighbours[..., np.newaxis].astype(np.float32)).astype(np.uint8)
        
        if depth_img is not None:
            new_depth = np.zeros((rgb_size[0], rgb_size[1]), dtype = np.float)
            new_depth[v[mask], u[mask]] = depth_img[mask]
            new_depth[intpol_mask] = (new_depth[up, left].astype(np.float32) + new_depth[up, right].astype(np.float32) + new_depth[down, left].astype(np.float32) \
                       + new_depth[down, right].astype(np.float32)) / count_unmasked_neighbours.astype(np.float32)
        else:
            new_depth = None

        if angle <= 0:
            crop = albumentations.Crop(x_min=0, y_min=0, x_max = crop_x, y_max = rows)
        else:
            crop = albumentations.Crop(x_min= crop_x, y_min= 0, x_max = cols, y_max = rows)

        if depth_img is not None:
            rgb_pipeline = albumentations.Compose([crop, resize], additional_targets={'depth': 'image'})
            transformed = rgb_pipeline(image=new_img, depth= new_depth)
            new_img = transformed["image"]
            new_depth = transformed["depth"]
        else:
            new_img_cropped = crop(image=new_img)["image"]
            new_img = resize(image=new_img_cropped)["image"]
            new_depth = None
        old_W = cols
        new_W = new_img.shape[1]
        crop_factor = 1 - (new_W / float(old_W))

        if angle > 0:
            intrinsics[0, 2] = intrinsics[0, 2] - (crop_factor * rgb_size[1])
        intrinsics[0, :] = intrinsics[0, :] * (1 / (1 - crop_factor))

        rot_center = (bev.shape[1] // 2, bev.shape[0])
        rot_mat = cv2.getRotationMatrix2D(rot_center, -angle, 1.0)
        bev_rotated = cv2.warpAffine(bev, rot_mat, (bev.shape[1], bev.shape[0]), flags=cv2.INTER_NEAREST, borderValue=self.ignore_index)

        fov_x = 2 * np.arctan(rgb_size[1] / (2 * intrinsics[0][0]))
        half_angle = fov_x /2
        line_len = (bev_rotated.shape[1] // 2 * np.sin(half_angle)) / np.sin(90 - half_angle)
        intersect = bev_rotated.shape[0] - line_len

        distance_z = np.arange(bev_rotated.shape[0])
        distance_z = np.expand_dims(distance_z, axis=1)
        mask = np.ones_like(bev_rotated) * (bev_rotated.shape[0] - 1) - distance_z
        if angle <= 0:
            mask[:, (bev_rotated.shape[1] // 2):] = -1
        else: 
            mask[:, :(bev_rotated.shape[1] // 2 + 1)] = -1
        mid_distance = np.abs(np.arange(bev_rotated.shape[1]) - bev_rotated.shape[1] // 2)
        mid_distance = np.expand_dims(mid_distance, axis = 0)
        mask = mask / (mid_distance + RotateAugmentation._EPSILON)
        mask_bool = np.logical_and(mask <= (intersect / (bev_rotated.shape[1] // 2)), mask >= 0)
        bev_rotated[mask_bool] = self.ignore_index

        weight_mask_rotated = cv2.warpAffine(weight_mask, rot_mat, (weight_mask.shape[1], weight_mask.shape[0]), flags=cv2.INTER_NEAREST, borderValue = 0)

        return new_img, bev_rotated, new_depth, weight_mask_rotated, intrinsics
