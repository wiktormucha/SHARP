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
# from config import action_to_verb, action_dict
import pandas as pd
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
import random
import warnings

import pytorchvideo.transforms.functional as T
import albumentations as A

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
                # Make it work only if it exists in transforms
               # ...existing code...
                if "replay" in transformed and "transforms" in transformed['replay']:
                    transforms = transformed['replay']["transforms"]
                    if "transforms" in transforms[0]:
                        horizontal_flip_flag = transforms[0]["applied"]
                        vertical_flip_flag = transforms[1]["applied"]
# ...existing code...

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
            # 'img_depth': img_depth_org
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


class H2O_actions(Dataset):

    def __init__(self, data_cfg, subset_type: str = 'train') -> None:
        super().__init__()

        self.using_own_pose = data_cfg.own_pose_flag

        self.data_dimension = data_cfg.data_dimension
        self.no_of_input_frames = data_cfg.no_of_input_frames
        annotation_pth = data_cfg.annotation_train
        data_for_model = data_cfg.data_for_model
        data_dir = data_cfg.data_dir
        self.using_2d_bb = data_cfg.hand_bb_flag
        self.using_obj_bb = data_cfg.using_obj_bb
        self.using_obj_label = data_cfg.using_obj_label
        self.subset_type = subset_type
        self.hand_pose_type = data_cfg.hand_pose_type
        self.obj_pose_type = data_cfg.obj_pose_type
        self.apply_vanishing = data_cfg.apply_vanishing
        self.vanishing_proability = data_cfg.vanishing_proability
        self.obj_to_vanish = data_cfg.obj_to_vanish

        if self.subset_type == 'train':
            self.sample = 'random'
            label_file = 'action_train.txt'

            self.own_obj_path = "/caa/Homes01/wmucha/repos/yolov7/h2o_bb_hands_train/exp/labels"
            self.path_to_own_pose = '/data/wmucha/datasets/h2o_ego/train_own_pose'
        elif self.subset_type == 'val':
            self.sample = 'uniform'
            label_file = 'action_val.txt'

            self.own_obj_path = "/caa/Homes01/wmucha/repos/yolov7/h2o_bb_hands/exp2/labels"
            self.path_to_own_pose = '/data/wmucha/datasets/h2o_ego/val_own_pose'
        elif self.subset_type == 'test':
            self.sample = 'uniform'
            label_file = 'action_test.txt'

            self.own_obj_path = "/caa/Homes01/wmucha/repos/yolov7/h2o_bb_hands_train/exp/labels"
            self.path_to_own_pose = '/data/wmucha/datasets/h2o_ego/train_own_pose'

        final_pth = os.path.join(annotation_pth, label_file)

        df = pd.DataFrame()
        if os.path.exists(final_pth):
            df = pd.read_csv(final_pth, sep=' ')
        else:
            raise ValueError("File not found: ", final_pth)

        self.id = df['id']
        self.path = df['path']
        if self.subset_type != 'test':
            self.action_label = df['action_label']
        self.start_act = df['start_act']
        self.end_act = df['end_act']
        self.start_frame = df['start_frame']
        self.end_frame = df['end_frame']
        self.data_path = data_dir
        self.width = 1280
        self.height = 720

        self.vid_clip_flag = False
        self.hand_pose_flag = False
        self.objs_flag = False
        self.frames_pths = []
        self.hand_pose_pths = []
        self.objs_pths = []

        vid_clip_flag = 'vid_clip' in data_for_model
        hand_pose_flag = 'hand_pose' in data_for_model
        objs_flag = 'obj' in data_for_model

        if vid_clip_flag:
            self.vid_clip_flag = True
            self.frames_pths = [self.__read_sequence_paths(
                init_pth=self.path[idx], start_idx=self.start_act[idx], end_idx=self.end_act[idx]) for idx in range(len(self.id))]

        if hand_pose_flag:
            self.hand_pose_flag = True
            self.hand_pose_pths = [self.__read_sequence_paths(init_pth=self.path[idx], start_idx=self.start_act[idx],
                                                              end_idx=self.end_act[idx], seq_type='hand_pose', file_format='.txt') for idx in range(len(self.id))]

        if objs_flag:
            self.objs_flag = True
            self.objs_pths = [self.__read_sequence_paths(init_pth=self.path[idx], start_idx=self.start_act[idx],
                                                         end_idx=self.end_act[idx], seq_type='obj_pose', file_format='.txt') for idx in range(len(self.id))]

    def __read_sequence_paths(self, init_pth, start_idx, end_idx, cam_no='cam4', seq_type='rgb', file_format='.png'):
        final_pth = os.path.join(self.data_path, init_pth, cam_no, seq_type)
        if not os.path.isdir(final_pth):
            raise ValueError("The path does not exist: ", final_pth)

        return [f'{final_pth}/{frame_id:06d}{file_format}' for frame_id in range(start_idx, end_idx+1)]

    def __load_handpose_to_tensor_no_albu(self, frames_list: np.array, obj_list: np.array, indxs_to_sample, cam_instr) -> torch.tensor:

        frames_list = frames_list[indxs_to_sample]
        obj_list = obj_list[indxs_to_sample]

        pts = []
        pts_z = []
        objs = []
        labels = []

        vanish_fag = False
        obj_to_vanish = None

        if (random.randint(0, 100) < (self.vanishing_proability*100)) & (self.apply_vanishing) and (self.subset_type == 'train'):
            vanish_fag = True
            obj_to_vanish = random.randint(0, 3)

        i = 0

        for frame, obj_pth in zip(frames_list, obj_list):

            if not os.path.isfile(frame):
                raise ValueError('File does not exist... ', frame)
            # Get hands
            if self.hand_pose_type == 'gt_hand_pose':

                hand_pose = np.loadtxt(frame)
                gt_pts = np.split(hand_pose, [1, 64, 65, 128])
                left_hand = np.reshape(gt_pts[1], (21, 3))
                right_hand = np.reshape(gt_pts[3], (21, 3))

                # Put to 2D

                is_left = int(gt_pts[0].tolist()[0])
                is_right = int(gt_pts[2].tolist()[0])

                merged3D = np.concatenate([left_hand, right_hand], axis=0)
                merged = project_points_3D_to_2D(merged3D, CAM_INTRS)
                hands_z = merged3D[:, 2].reshape(42, 1)

                if is_left == 0:
                    merged[:21] = 0
                    hands_z[:21] = 0

                if is_right == 0:
                    merged[21:] = 0
                    hands_z[21:] = 0

            elif self.hand_pose_type == 'hand_pose_3d_own':

                hand_pose = np.loadtxt(frame.replace('subject1', 'subject1_ego').replace('subject2', 'subject2_ego').replace('subject3', 'subject3_ego').replace('subject4', 'subject4_ego').replace(
                    'hand_pose', 'hand_pose_3d_own'))

                hand_pose = hand_pose.reshape(42, 3)
                hand_pose[:, 2] = hand_pose[:, 2] / 1000

                merged = hand_pose[:, 0:2]
                hands_z = hand_pose[:, 2].reshape(42, 1)

            elif self.hand_pose_type == 'hand_pose_3d_own_not_masked':

                hand_pose = np.loadtxt(frame.replace('subject1', 'subject1_ego').replace('subject2', 'subject2_ego').replace('subject3', 'subject3_ego').replace('subject4', 'subject4_ego').replace(
                    'hand_pose', 'hand_pose_3d_own_not_masked'))

                hand_pose = hand_pose.reshape(42, 3)
                hand_pose[:, 2] = hand_pose[:, 2] / 1000

                merged = hand_pose[:, 0:2]
                hands_z = hand_pose[:, 2].reshape(42, 1)

            elif self.hand_pose_type == 'hand_pose_3d_own_masked':

                hand_pose = np.loadtxt(frame.replace('subject1', 'subject1_ego').replace('subject2', 'subject2_ego').replace('subject3', 'subject3_ego').replace('subject4', 'subject4_ego').replace(
                    'hand_pose', 'hand_pose_3d_own_masked'))

                hand_pose = hand_pose.reshape(42, 3)
                hand_pose[:, 2] = hand_pose[:, 2] / 1000

                merged = hand_pose[:, 0:2]
                hands_z = hand_pose[:, 2].reshape(42, 1)

            elif self.hand_pose_type == 'own_hand_pose':

                hand_pose = np.loadtxt(frame.replace('subject1', 'subject1_ego').replace('subject2', 'subject2_ego').replace('subject3', 'subject3_ego').replace('subject4', 'subject4_ego').replace(
                    'hand_pose', 'hand_pose_ownmodel'))
                merged = hand_pose.reshape(42, 2)

            elif self.hand_pose_type == 'mediapipe_hand_pose':

                hand_pose = np.loadtxt(frame.replace('subject1', 'subject1_ego').replace('subject2', 'subject2_ego').replace('subject3', 'subject3_ego').replace('subject4', 'subject4_ego').replace(
                    'hand_pose', 'hand_pose_mediapipe'))
                merged = hand_pose.reshape(42, 2)

            elif self.hand_pose_type == 'ego_handpoints':

                hand_pose = np.loadtxt(frame.replace('subject1', 'subject1_ego').replace('subject2', 'subject2_ego').replace('subject3', 'subject3_ego').replace('subject4', 'subject4_ego').replace(
                    'hand_pose', 'hand_pose_ownmodel_ego'))
                merged = hand_pose.reshape(42, 2)

            elif self.hand_pose_type == 'hand_resnet50':

                hand_pose = np.loadtxt(frame.replace('subject1', 'subject1_ego').replace('subject2', 'subject2_ego').replace('subject3', 'subject3_ego').replace('subject4', 'subject4_ego').replace(
                    'hand_pose', 'hand_pose_resnet50'))
                merged = hand_pose.reshape(42, 2)

            kpts2d_img = merged.copy()
            # Getting OBJ label
            if self.using_obj_bb or self.using_obj_label:

                obj_pose = np.loadtxt(obj_pth)
                pts_3d = obj_pose[1:64].reshape((21, 3))
                obj_label = obj_pose[0]

                if self.using_obj_bb:

                    if self.obj_pose_type == 'GT':
                        pts_2d = project_points_3D_to_2D(pts_3d, cam_instr)
                        xmax, ymax = pts_2d.max(axis=0)
                        xmin, ymin = pts_2d.min(axis=0)
                        obj_bb = np.array([xmin, ymin,
                                           xmax, ymin, xmin, ymax, xmax, ymax])

                        obj_bb = obj_bb.reshape(4, 2)

                    elif self.obj_pose_type == 'YoloV7':
                        # TODO load yolo obj
                        objs_key_list = [0, 1, 2, 3, 4, 5, 6, 7]
                        obj_pose_pth = frame.replace('subject1', 'subject1_ego').replace('subject2', 'subject2_ego').replace('subject3', 'subject3_ego').replace('subject4', 'subject4_ego').replace(
                            'hand_pose', 'obj_pose_ownmodel')

                        yolo_labels_file = pd.read_csv(
                            obj_pose_pth, sep=" ", header=None, index_col=None)
                        yolo_objs = read_yolo_labels(
                            yolo_labels=yolo_labels_file)

                        obj_bb = 0
                        for key in yolo_objs:
                            if key in objs_key_list:
                                obj = yolo_objs[key][0]

                                ymin = ((obj.yc - (obj.height/2)))
                                ymax = ((obj.yc + (obj.height/2)))
                                xmin = ((obj.xc - (obj.width/2)))
                                xmax = ((obj.xc + (obj.width/2)))

                                obj_bb = np.array([xmin, ymin,
                                                   xmax, ymin, xmin, ymax, xmax, ymax])

                                obj_bb = obj_bb.reshape(
                                    4, 2) * (self.width, self.height)

                                obj_label = (obj.label + 1)
                        if type(obj_bb) is int:

                            obj_bb = np.zeros((4, 2))

            pts.append(torch.tensor(kpts2d_img))
            pts_z.append(torch.tensor(hands_z))
            objs.append(torch.tensor(obj_bb))
            labels.append(torch.tensor(obj_label))

            i += 1

        return np.array(pts), np.array(pts_z), np.array(objs), np.array(labels), vanish_fag, obj_to_vanish

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):

        action_id = self.id[idx]

        if self.subset_type == "test":
            action = 999
        else:
            action = self.action_label[idx]

        if self.hand_pose_flag:

            indxs_to_sample = sample2(
                input_frames=self.hand_pose_pths[idx], no_of_outframes=self.no_of_input_frames, sampling_type=self.sample)

            pts_2d_img, pts_z, objs_img, labels, vanish_flag, obj_to_vanish = self.__load_handpose_to_tensor_no_albu(
                frames_list=np.array(self.hand_pose_pths[idx]), obj_list=np.array(self.objs_pths[idx]), indxs_to_sample=indxs_to_sample, cam_instr=CAM_INTRS)

            # Make random choice between 0 and 1
            # augument_data = False
            augument_data = random.randint(0, 1)

            if augument_data and self.subset_type == 'train' and vanish_flag == False:

                theta = random.randint(-45, 45)
                pts_2d_img = rotate_keypoints(
                    pts_2d_img, self.width, self.height, theta)
                objs_img = rotate_keypoints(
                    objs_img, self.width, self.height, theta)

            positions = concatenate_and_normalise(
                pts_2d_img, pts_z, objs_img, labels, self.width, self.height)

            if vanish_flag and self.subset_type == 'train':
                positions = vanish_keypoints_sequence(positions, obj_to_vanish)

        return {
            "action_id": action_id,
            "action_label": action - 1,
            "positions": positions,
        }


def flip_horizontaly_kpts_seq(kpts, img_width):
    fliiped_kpts = kpts.copy()
    fliiped_kpts[::2] = img_width - fliiped_kpts[::2]
    return fliiped_kpts


def rotate_keypoints(kpts_tensor, image_width, image_height, theta):
    # Convert angle to radians
    theta = np.radians(theta)

    # Calculate the center of the image
    center_x = image_width / 2.0
    center_y = image_height / 2.0

    # Shift the coordinates so that the center of the image is at (0,0)
    shifted_kpts = kpts_tensor.copy()
    shifted_kpts[:, :, 0] -= center_x
    shifted_kpts[:, :, 1] -= center_y

    # Apply the rotation matrix
    rotated_kpts = shifted_kpts.copy()
    rotated_kpts[:, :, 0] = shifted_kpts[:, :, 0] * \
        np.cos(theta) - shifted_kpts[:, :, 1]*np.sin(theta)
    rotated_kpts[:, :, 1] = shifted_kpts[:, :, 0] * \
        np.sin(theta) + shifted_kpts[:, :, 1]*np.cos(theta)

    # Shift the coordinates back
    rotated_kpts[:, :, 0] += center_x
    rotated_kpts[:, :, 1] += center_y

    return rotated_kpts


def concatenate_and_normalise(pts_2d_img, pts_z, objs_img, labels, width, height):

    pts_2d_norm = pts_2d_img / (width, height)
    objs_norm = objs_img / (width, height)
    objs_norm = objs_norm.reshape(objs_norm.shape[0], -1)

    kpts25_norm = np.concatenate([pts_2d_norm, pts_z], axis=2).reshape(
        pts_2d_img.shape[0], -1)

    merged = np.concatenate([kpts25_norm, objs_norm], axis=1)
    merged = np.concatenate(
        [merged, labels.reshape(objs_norm.shape[0], 1)], axis=1)

    return merged


def vanish_keypoints_sequence(keypoints, obj_to_vanish):
    # type = 2
    if obj_to_vanish == 0:
        # Make left hand equal to 0
        keypoints[:, 0:63] = 0
    elif obj_to_vanish == 1:
        # Make right hand equal to 0
        keypoints[:, 63:126] = 0
    elif obj_to_vanish == 2:
        # Make obj pose equal to 0
        keypoints[:, 126:134] = 0
    elif obj_to_vanish == 3:
        # Make obj label equal to 0
        keypoints[:, -1] = 0

    return keypoints


def switch_left_with_right(pts_2d_img):
    ptsL = pts_2d_img[:, 0:21]
    ptsR = pts_2d_img[:, 21:]

    pts_2d_img = np.concatenate([ptsR, ptsL], axis=1)
    return pts_2d_img


def sample2(input_frames: list, no_of_outframes, sampling_type: str):

    if len(input_frames) >= no_of_outframes:
        indxs_to_sample = np.arange(len(input_frames))

        if sampling_type == "uniform":

            # Uniformly susample the frames to match the no_of_outframes
            # indxs_to_sample = np.linspace(
            #     0, len(input_frames)-1, no_of_outframes, dtype=int)

            indxs_to_sample = T.uniform_temporal_subsample(
                torch.tensor(indxs_to_sample), no_of_outframes, 0).tolist()

        elif sampling_type == "random":

            # randomly susample the frames to match the no_of_outframes
            indxs_to_sample = list(range(len(input_frames)))
            indxs_to_sample = random.sample(
                indxs_to_sample, no_of_outframes)
            indxs_to_sample.sort()

    else:
        indxs_to_sample = np.trunc(
            np.arange(0, no_of_outframes) * len(input_frames)/no_of_outframes).astype(int)

    return indxs_to_sample


class YoloLabel:
    label: int
    xc: float
    yc: float
    width: float
    height: float


def read_yolo_labels(yolo_labels: pd.DataFrame, bb_factor=1):
    ret_dict = {}
    for idx, row in yolo_labels.iterrows():
        temp_obj = YoloLabel()
        # row = row[0]
        temp_obj.label = int(row[0])
        temp_obj.xc = float(row[1])
        temp_obj.yc = float(row[2])
        temp_obj.width = float(row[3]) * bb_factor
        temp_obj.height = float(row[4]) * bb_factor

        if temp_obj.label not in ret_dict:
            ret_dict[temp_obj.label] = [temp_obj]
        else:
            ret_dict[temp_obj.label].append(temp_obj)

    return ret_dict
