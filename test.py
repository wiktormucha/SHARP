from utils.trainer import Trainer3D
from spock_dataclasses import *
from albumentations.pytorch.transforms import ToTensorV2
from utils.general_utils import make_optimiser
from utils import loses
from utils.general_utils import freeze_seeds, set_max_cpu_threads
from datasets.h2o import get_h2o_dataloaders
from spock import SpockBuilder
import albumentations as A
from models.models import make_model

def main() -> None:

    set_max_cpu_threads()
    freeze_seeds()

    # Build Spock Config
    cfg = SpockBuilder(H2oHandsData, TrainingConfig,
                       desc='Quick start example').generate()

    albumentation_val = A.ReplayCompose(
        [
            A.Resize(cfg.H2oHandsData.img_size[0],
                     cfg.H2oHandsData.img_size[1]),
            A.Normalize(mean=cfg.H2oHandsData.norm_mean,
                        std=cfg.H2oHandsData.norm_std),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )

    dataloaders = get_h2o_dataloaders(
        h2o_cfg=cfg.H2oHandsData, num_workers=6, albu_val=albumentation_val, albu_test=albumentation_val)
    val_dataloader = dataloaders['val']
    test_dataloader = dataloaders['test']

    model = make_model(model_cfg=cfg.TrainingConfig,
                       device=cfg.TrainingConfig.device)

    criterion = getattr(loses, cfg.TrainingConfig.criterion)()
    optimiser = make_optimiser(model=model, training_cfg=cfg.TrainingConfig)
    trainer = Trainer3D(model, criterion, optimiser,
                        cfg.TrainingConfig, wandb_logger=None)

    print('Testing model on validation:')
    trainer.test(val_dataloader)
    print('Testing model on test:')
    trainer.test(test_dataloader)


if __name__ == '__main__':
    main()
