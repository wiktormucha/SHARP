from utils.trainer import Trainer3D
import torch
import sys
from spock_dataclasses import *
import wandb
import yaml
from albumentations.pytorch.transforms import ToTensorV2
from utils.general_utils import make_model, make_optimiser, freeze_seeds, set_max_cpu_threads
from utils import loses
from datasets.h2o import get_h2o_dataloaders


def main() -> None:

    # Build Spock Config
    cfg = SpockBuilder(H2oHandsData, TrainingConfig,
                       desc='Quick start example').generate()

    # Set up fixed shuffeling for hyperparameter tuning
    freeze_seeds(cfg.TrainingConfig.seed)
    set_max_cpu_threads()

    wandbcfg_pth = sys.argv[2]
    # opening a file
    with open(wandbcfg_pth, 'r') as stream:
        try:
            # Converts yaml document to python object
            wandbcfg = yaml.safe_load(stream)
        # Program to convert yaml file to dictionary
        except yaml.YAMLError as e:
            print(e)

    albumentations_train = A.ReplayCompose(
        [
            A.OneOf([
                    A.HorizontalFlip(always_apply=False, p=0.33),
                    A.VerticalFlip(always_apply=True, p=0.33),
                    A.Compose([
                        A.HorizontalFlip(always_apply=True, p=1.0),
                        A.VerticalFlip(always_apply=True, p=1.0),
                    ], p=0.33)
                    ], p=0.6),
            A.OneOf([
                    A.Resize(
                        cfg.H2oHandsData.img_size[0], cfg.H2oHandsData.img_size[1]),
                    A.RandomResizedCrop(always_apply=True, p=0.5, height=cfg.H2oHandsData.img_size[0],
                                        width=cfg.H2oHandsData.img_size[1], scale=(0.7, 1.0), ratio=(1, 1), interpolation=0),
                    A.Compose([
                        A.Rotate(always_apply=True, p=0.5, limit=(-30, 30), interpolation=0, border_mode=4,
                                 value=(0, 0, 0), mask_value=None, rotate_method='largest_box', crop_border=False),
                        A.RandomResizedCrop(always_apply=True, p=0.5, height=cfg.H2oHandsData.img_size[0],
                                            width=cfg.H2oHandsData.img_size[1], scale=(0.7, 1.0), ratio=(1, 1), interpolation=0),
                    ]),
                    ], p=1.0),

            A.Normalize(mean=cfg.H2oHandsData.norm_mean,
                        std=cfg.H2oHandsData.norm_std),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )

    albumentation_val = A.ReplayCompose(
        [
            A.Resize(cfg.H2oHandsData.img_size[0],
                     cfg.H2oHandsData.img_size[1]),
            A.Normalize(mean=cfg.H2oHandsData.norm_mean,
                        std=cfg.H2oHandsData.norm_std),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )

    wandbcfg['albu_train'] = albumentations_train
    wandbcfg['albu_val'] = albumentation_val

    dataloaders = get_h2o_dataloaders(
        h2o_cfg=cfg.H2oHandsData, num_workers=6, albu_train=albumentations_train, albu_val=albumentation_val)

    train_dataloader = dataloaders['train']
    val_dataloader = dataloaders['val']

    if cfg.TrainingConfig.debug:
        logger = None
    else:
        # set the wandb project where this run will be logged
        logger = wandb.init(project=cfg.TrainingConfig.experiment_name,
                            config=wandbcfg)

    model = make_model(model_cfg=cfg.TrainingConfig,
                       device=cfg.TrainingConfig.device)

    criterion = getattr(loses, cfg.TrainingConfig.criterion)()
    optimiser = make_optimiser(model=model, training_cfg=cfg.TrainingConfig)

    # Check if dataset is H2O or AssemblyHands
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimiser, milestones=[50, 60, 70, 80, 90, 100], gamma=0.5, last_epoch=- 1, verbose=True)

    trainer = Trainer3D(model, criterion, optimiser,
                        cfg.TrainingConfig, wandb_logger=logger, grad_clip=cfg.TrainingConfig.grad_clipping, scheduler=scheduler)

    print(f'Starting training on device: {cfg.TrainingConfig.device}')

    model = trainer.train(
        train_dataloader=train_dataloader, val_dataloader=val_dataloader)

    if not cfg.TrainingConfig.debug:
        wandb.finish()


if __name__ == '__main__':
    main()
