from typing import List
from spock import spock
from enum import Enum
from typing import List


class DepthType(Enum):
    est = 'est'
    gt = 'gt'
    est_low = 'est_low'


class Model(Enum):
    effhandegonet3d = 'EffHandEgoNet3D'


@spock
class TrainingConfig:

    seed: int = 42
    debug: bool = False
    device: int
    lr: float
    max_epochs: int
    early_stopping: int
    weight_decay: float
    momentum: float
    load_model: bool
    load_model_path: str
    model_type: Model
    experiment_name: str
    criterion: str
    num_workers: int


@spock
class H2oHandsData:
    path: str
    imgs_path: str
    batch_size: int
    img_size: List[int]
    max_train_samples: int = 55742
    max_val_samples: int = 11638
    norm_mean: List[float]
    norm_std: List[float]
    use_depth: bool = False
    depth_img_type: DepthType
    segment_list: List[int]
    segment_list_val: List[int]
