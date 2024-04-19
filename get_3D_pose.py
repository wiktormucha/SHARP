

from datasets.h2o import get_h2o_dataloaders
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from spock_dataclasses import *
from tqdm import tqdm
import os
from tqdm import tqdm
import torch.nn as nn
from models.models import make_model
import albumentations as A
import argparse


def main():

    parser = argparse.ArgumentParser(description='Arguments to run a model')

    parser.add_argument('--path', type=str, default="/data/wmucha/datasets",
                        help='path to the dataset')
    parser.add_argument('--imgs_path', type=str, default="h2o_ego/h2o_ego",
                        help='path to the folder with images in the dataset folder')
    parser.add_argument('--use_depth', type=bool, default=True,
                        help='Do you use depth images? - required for SHARP')
    parser.add_argument('--device', type=int, default=2,
                        help='device to run the model')
    parser.add_argument('--load_model', type=bool, default=True,
                        help='Do you want to load model weights')
    parser.add_argument('--load_model_path', type=str, default='checkpoints/fluent-silence-280/checkpoint_best.pth',
                        help='Path to model weights')
    args = parser.parse_args()

    config = H2oHandsData(path=args.path,
                          imgs_path=args.imgs_path,
                          batch_size=1,
                          img_size=[512, 512],
                          norm_mean=[0.485, 0.456, 0.406],
                          norm_std=[0.229, 0.224, 0.225],
                          use_depth=args.use_depth,
                          depth_img_type='est_low',
                          segment_list=[0, 120],
                          segment_list_val=[120]
                          )

    train_cfg = TrainingConfig(

        experiment_name='3DHPDSEG',
        device=args.device,
        lr=0.1,
        max_epochs=500,
        early_stopping=13,
        weight_decay=0.0,
        momentum=0.9,
        criterion='EffHandEgoNetLoss3D',
        num_workers=6,
        load_model=args.load_model,
        load_model_path=args.load_model_path,
        model_type='EffHandEgoNet3D',
    )

    albumentation_val = A.Compose(
        [
            A.Resize(512,
                     512),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )

    dataloaders = get_h2o_dataloaders(
        config, albu_train=albumentation_val, albu_val=albumentation_val, albu_test=albumentation_val)

    train_dataloader = dataloaders['train']
    val_dataloader = dataloaders['val']
    test_dataloader = dataloaders['test']

    dataloaders_lst = [train_dataloader, val_dataloader, test_dataloader]

    if config.use_depth:
        folder_to_save = "hand_pose_3d_own_masked"
    else:
        folder_to_save = "hand_pose_3d_own_not_masked"

    model = make_model(model_cfg=train_cfg,
                       device=train_cfg.device)

    model.eval()

    activation = nn.Sigmoid()

    for dataloader in dataloaders_lst:
        print(
            f'Predicting poses for {str(dataloader.dataset.subset_type)} subset')
        for i, batch in enumerate(tqdm(dataloader)):
            img_path = batch['img_path'][0]

            img_path = img_path.split('/')
            file_name = img_path[-1].replace('.png', '.txt')
            folder_path = os.path.join('/data/wmucha/datasets/h2o_ego/h2o_ego',
                                       img_path[6], img_path[7], img_path[8], img_path[9], folder_to_save)

            file_path = os.path.join(folder_path, file_name)

            # check if file exists
            if os.path.exists(file_path):
                continue

            # Check if folder exists
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Predict pose:
            with torch.no_grad():
                outs = model(batch['img'].to(train_cfg.device))

            pred_kpts3d = outs['kpts_3d_cam'].cpu().detach().numpy()
            pred_kpts3d_img = outs['kpts25d'].cpu().detach().numpy()

            left_batch_mask = int(torch.argmax(
                activation(outs['left_handness']), dim=1).cpu().detach().numpy())
            right_batch_mask = int(torch.argmax(
                activation(outs['right_handness']), dim=1).cpu().detach().numpy())

            pred_kpts3d[:, :21, :] = pred_kpts3d[:, :21, :] * left_batch_mask
            pred_kpts3d[:, 21:, :] = pred_kpts3d[:, 21:, :] * right_batch_mask

            np.savetxt(file_path, pred_kpts3d_img.reshape(-1, 63))


if __name__ == '__main__':
    main()
