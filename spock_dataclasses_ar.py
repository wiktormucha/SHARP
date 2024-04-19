from typing import List
from spock import spock
from spock import SpockBuilder
from models import models
from enum import Enum
from typing import List
from typing import Optional
from typing import Tuple
import torch.optim as optim
import albumentations as A


class HandPoseType(Enum):
    gt_hand_pose = 'gt_hand_pose'
    hand_pose_3d_own_masked = 'hand_pose_3d_own_masked'
    hand_pose_3d_own_not_masked = 'hand_pose_3d_own_not_masked'


class ObjPoseType(Enum):
    gt = 'GT'
    yolov7 = 'YoloV7'


class DataDimension(Enum):
    two_d = '2D'
    thre_d = '3D'


class Optimizer(Enum):
    sgd = 'SGD'
    adam = 'Adam'
    adamw = 'AdamW'

class Model(Enum):
    Hand3DActionTransformer = 'Hand3DActionTransformer'


@spock
class ModelConfig():
    model_type: Model
    input_dim: int
    hidden_layers: int
    out_dim: int
    dropout: float
    dropout_att: float
    trans_num_layers: int
    trans_num_heads: int
    seq_length: int
    load_checkpoint: bool
    checkpoint_path: str
    dropout_mlp: float


@spock
class OptimizerConfig:
    type: Optimizer = 'SGD'
    lr: float = 0.01
    weight_decay: Optional[float] = 0.0
    momentum: Optional[float] = 0.0


@spock
class TrainingConfig:
    scheduler_milestones: List[int]
    seed_num: int
    max_epochs: int
    batch_size: int
    device: int
    early_stopping: int
    emplying_objs: bool = False
    checkpoint_pth: str


@spock
class DataConfig:
    data_dir: str = 'Asdf'
    annotation_train: str
    data_dimension: DataDimension
    data_for_model: List[str]
    no_of_input_frames: int
    hand_bb_flag: bool
    using_obj_bb: bool
    own_pose_flag: bool
    using_obj_label: bool
    hand_pose_type: HandPoseType
    obj_pose_type: ObjPoseType
    apply_vanishing: bool
    vanishing_proability: float
    obj_to_vanish: int
