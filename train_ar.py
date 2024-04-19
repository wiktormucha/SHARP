from datasets.h2o import H2O_actions
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spock_dataclasses_ar import *
from spock import SpockBuilder
from utils.general_utils import freeze_seeds
from utils.trainer import TrainerAR
from utils.general_utils import define_optimizer
import wandb
import yaml
import sys


def main() -> None:
    """
    Main training loop
    """

    # Build config
    config = SpockBuilder(OptimizerConfig, ModelConfig, TrainingConfig, DataConfig,
                          desc='Quick start example').generate()

    freeze_seeds(seed_num=config.TrainingConfig.seed_num)

    # Load config yaml to wandb
    wandbcfg_pth = sys.argv[2]
    # opening a file
    with open(wandbcfg_pth, 'r') as stream:
        try:
            # Converts yaml document to python object
            wandbcfg = yaml.safe_load(stream)

        # Program to convert yaml file to dictionary
        except yaml.YAMLError as e:
            print(e)

    logger = wandb.init(
        # set the wandb project where this run will be logged
        project="h2o_action_recognition_3d_pose",
        config=wandbcfg)

    train_dataset = H2O_actions(
        data_cfg=config.DataConfig)

    print("Len of train: ", len(train_dataset))

    train_dataloader = DataLoader(
        train_dataset,
        config.TrainingConfig.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=12,
        pin_memory=True,

    )

    val_dataset = H2O_actions(data_cfg=config.DataConfig,
                              subset_type="val")

    print("Len of val: ", len(val_dataset))

    val_dataloader = DataLoader(
        val_dataset,
        config.TrainingConfig.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=12,
        pin_memory=True,

    )

    # Create model
    model = getattr(models, config.ModelConfig.model_type)(
        config.ModelConfig, device=config.TrainingConfig.device)
    model = model.to(config.TrainingConfig.device)

    # If loading weights from checkpoin
    if config.ModelConfig.load_checkpoint:
        model.load_state_dict(torch.load(
            config.ModelConfig.checkpoint_path, map_location=torch.device(config.TrainingConfig.device)))
        print("Model's checkpoint loaded")

    criterion = nn.CrossEntropyLoss()
    optimizer = define_optimizer(model, config.OptimizerConfig)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.TrainingConfig.scheduler_milestones, gamma=0.5, last_epoch=- 1, verbose=True)

    trainer = TrainerAR(model, criterion, optimizer,
                        config.TrainingConfig, wandb_logger=logger, scheduler=scheduler)
    print(f'Starting training on device: {config.TrainingConfig.device}')

    model = trainer.train(train_dataloader, val_dataloader)

    wandb.finish()


if __name__ == '__main__':
    main()
