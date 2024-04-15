import warnings
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from utils.general_utils import project_points_3D_to_2D, vector_to_heatmaps
import cv2 as cv2
import random
from torch.utils.data import DataLoader

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=np.VisibleDeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

CAM_INTRS = np.array([[636.6593017578125, 0.00000000e+00, 635.283881879317],
                      [0.00000000e+00, 636.251953125, 366.8740353496978],
                      [0.00000000e+00, 0.00000000e+00, 1.0]])


class H2O_Dataset_hand_train_3D(Dataset):
    """
    H2O Dataset for egocentric hand tests only
    """

    def __init__(self, config, type="train", albu_transform=None):
        self.subset_type = type
        type_mapping = {
            "train": ("train_pose.txt", config.max_train_samples),
            "val": ("val_pose.txt", config.max_val_samples),
            "test": ("test_pose.txt", 30000)
        }
        path_list, max_samples = type_mapping.get(
            type, ("train_pose.txt", config.max_train_samples))
        path_list = os.path.join(config.path, path_list)

        self.imgs_paths = pd.read_csv(path_list, header=None).to_numpy()[
            0:max_samples]
        self.img_size = config.img_size
        self.albu_transform = albu_transform

        self.depth_img_type = config.depth_img_type
        self.use_depth = config.use_depth
        if self.subset_type == "train":
            self.segment_list = config.segment_list
        else:
            self.segment_list = config.segment_list_val

        self.heatmap_dim = 56 if self.img_size[0] == 224 else 128 if self.img_size[0] == 512 else None

        base_path = os.path.join(config.path, config.imgs_path)
        self.imgs_paths = [os.path.join(base_path, path[0])
                           for path in self.imgs_paths]

        hand_pose_pths = [path.replace("rgb", "hand_pose").replace(
            ".png", ".txt") for path in self.imgs_paths]

        hand_pose_list, hand_pose3D_list, left_hand_flag_temp_list, right_hand_flag_temp_list = [], [], [], []

        for path in hand_pose_pths:
            hand_pose, hand_pose3D, left_hand_flag_temp, right_hand_flag_temp = self.__read_hand_pose(
                path, CAM_INTRS)
            hand_pose_list.append(hand_pose)
            hand_pose3D_list.append(hand_pose3D)
            left_hand_flag_temp_list.append(left_hand_flag_temp)
            right_hand_flag_temp_list.append(right_hand_flag_temp)

        self.hand_pose2D = np.array(hand_pose_list)
        self.hand_pose3D = np.array(hand_pose3D_list)
        self.left_hand_flag = np.array(left_hand_flag_temp_list)
        self.right_hand_flag = np.array(right_hand_flag_temp_list)

    def __len__(self) -> int:
        """
        Return length of the dataset

        Returns:
            int: Dataset length
        """
        return len(self.imgs_paths)

    def __read_hand_pose(self, path, cam_instr):
        hand_pose = np.loadtxt(path)
        gt_pts = np.split(hand_pose, [1, 64, 65, 128])
        left_hand = gt_pts[1].reshape((21, 3))
        right_hand = gt_pts[3].reshape((21, 3))
        keypoints3d = np.concatenate([left_hand, right_hand])
        keypoints2d = project_points_3D_to_2D(keypoints3d, cam_instr)
        left_hand_flag = gt_pts[0]
        right_hand_flag = gt_pts[2]

        return keypoints2d, keypoints3d, left_hand_flag, right_hand_flag

    def __getitem__(self, idx: int) -> dict:
        return self._getitem_frame(idx)

    def _getitem_frame(self, idx: int) -> dict:

        img_path = self.imgs_paths[idx]
        img = np.array(Image.open(img_path))

        if self.use_depth:
            # Use orginal depth image
            if self.depth_img_type == "gt":
                img_path_depth = img_path.replace('rgb', 'depth')
                img_depth = cv2.imread(img_path_depth, cv2.IMREAD_ANYDEPTH)
                img_depth = img_depth / 1.0
                img_depth_org = img_depth.copy()

            # Use pseudo_depth image
            elif self.depth_img_type == "est_low":
                # print("Using low res depth")
                img_path_depth_est = img_path.replace('rgb', 'depth_est_low')
                img_depth = cv2.imread(img_path_depth_est, cv2.IMREAD_ANYDEPTH)
                img_depth = cv2.resize(img_depth, (1280, 720))
                img_depth_org = img_depth.copy()

            # Choose random treshold and apply
            treshold = random.choice(self.segment_list)
            img_masked = segment_background(
                img=img, img_depth=img_depth, threshold=treshold, depth_type=self.depth_img_type)
            img = img_masked

        hand_pose = self.hand_pose2D[idx]

        # Meters to milimeters
        pose_3d_gt = self.hand_pose3D[idx] * 1000
        left_hand_flag_temp = self.left_hand_flag[idx]
        right_hand_flag_temp = self.right_hand_flag[idx]

        horizontal_flip_flag = False
        vertical_flip_flag = False
        if self.albu_transform:

            transformed = self.albu_transform(
                image=img, keypoints=hand_pose)

            img = transformed['image']
            keypoints = np.array(transformed['keypoints'])

            if self.subset_type == "train":
                transforms = transformed['replay']["transforms"][0]["transforms"]
                horizontal_flip_flag = transforms[0]["applied"]
                vertical_flip_flag = transforms[1]["applied"]

            else:
                horizontal_flip_flag = False
                vertical_flip_flag = False

        else:
            img = img
            keypoints = hand_pose

        # If image was mirroed than switch hands kpts
        if horizontal_flip_flag or vertical_flip_flag:

            right_hand_flag = left_hand_flag_temp
            left_hand_flag = right_hand_flag_temp

            ptsL = keypoints[21:42]
            ptsR = keypoints[0:21]

            # concatenate ptsL and ptsR to shape (42,2)
            keypoints = np.concatenate((ptsL, ptsR), axis=0)

            ptsL_3D = self.hand_pose3D[idx][21:42]
            ptsR_3D = self.hand_pose3D[idx][0:21]
            pose_3d_gt = np.concatenate((ptsL_3D, ptsR_3D), axis=0)
        else:

            left_hand_flag = left_hand_flag_temp
            right_hand_flag = right_hand_flag_temp

        # Multiple first 21 values in keypoints of shape (42,2) by flag_1 and second 21 by flag_2
        keypoints[:21] *= left_hand_flag
        keypoints[21:] *= right_hand_flag
        pose_3d_gt[:21] *= left_hand_flag
        pose_3d_gt[21:] *= right_hand_flag

        heatmaps = vector_to_heatmaps(
            keypoints/self.img_size, scale_factor=self.heatmap_dim, out_size=self.heatmap_dim, n_keypoints=42)
        heatmaps[:21] *= left_hand_flag
        heatmaps[21:] *= right_hand_flag

        return {
            "img_path": img_path,
            "img": img,
            "left_hand_flag": int(left_hand_flag),
            "right_hand_flag": int(right_hand_flag),
            "keypoints": keypoints / self.img_size,
            'kpts_3d_cam': pose_3d_gt,
            'heatmaps': heatmaps,
            'kpts_2d_img': keypoints / self.img_size * (1280, 720),
            'img_depth': img_depth_org
        }


def segment_background(img: np.array, img_depth: np.array, threshold: int, depth_type) -> np.array:

    if threshold == 0:
        return img

    img_masked = np.zeros_like(img)

    if depth_type == "est" or depth_type == "est_low":
        img_depth[img_depth < threshold] = 0
    elif depth_type == "gt":
        img_depth[img_depth > threshold] = 0

    # Blur Image
    # img_depth = cv2.GaussianBlur(img_depth.astype(np.float32), (61, 61), 100)

    img_masked[img_depth != 0] = img[img_depth != 0]

    return img_masked


def get_h2o_dataloaders(h2o_cfg, num_workers=6, albu_train=None, albu_val=None, albu_test=None):

    ret_dict = {}

    if albu_train:
        train_dataset = H2O_Dataset_hand_train_3D(
            config=h2o_cfg, type='train', albu_transform=albu_train)

        train_dataloader = DataLoader(
            train_dataset,
            h2o_cfg.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        ret_dict['train'] = train_dataloader

    if albu_val:
        val_dataset = H2O_Dataset_hand_train_3D(
            config=h2o_cfg, type='val', albu_transform=albu_val)

        val_dataloader = DataLoader(
            val_dataset,
            h2o_cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        ret_dict['val'] = val_dataloader

    if albu_test:
        test_dataset = H2O_Dataset_hand_train_3D(
            config=h2o_cfg, type='test', albu_transform=albu_test)

        test_dataloader = DataLoader(
            test_dataset,
            h2o_cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True
        )

        ret_dict['test'] = test_dataloader

    return ret_dict
