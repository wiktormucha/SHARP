from datasets.h2o import H2O_actions
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from spock_dataclasses_ar import *
import importlib.util
from spock import SpockBuilder
from utils.general_utils import freeze_seeds
from utils.trainer import TrainerAR
from utils.general_utils import define_optimizer


def main() -> None:
    """
    Main training loop
    """

    # Build config
    config = SpockBuilder(OptimizerConfig, ModelConfig, TrainingConfig, DataConfig,
                          desc='Quick start example').generate()

    freeze_seeds(seed_num=config.TrainingConfig.seed_num)

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

    test_dataset = H2O_actions(data_cfg=config.DataConfig,
                               subset_type="test")

    print("Len of val: ", len(test_dataset))

    test_dataloader = DataLoader(
        test_dataset,
        config.TrainingConfig.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=12,
        pin_memory=True,

    )

    print("Len of test: ", len(test_dataset))

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
                        config.TrainingConfig, model_config=config.ModelConfig, scheduler=scheduler)
    print(f'Starting training on device: {config.TrainingConfig.device}')

    model = trainer.test_model(val_dataloader)
    model = trainer.test_h2o(test_dataloader)


if __name__ == '__main__':
    main()
