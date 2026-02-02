from omegaconf import OmegaConf
from dataclasses import dataclass, field
from typing import List, Literal, Union, Optional


@dataclass
class Model:
    architecture: str
    pretrained: Optional[bool] = True
    checkpoint: Optional[str] = None
    input_channel: Optional[int] = 3
    dropout_rate: Optional[float] = 0.1
    size_factor: Optional[float] = 1.0
    single_cls: Optional[bool] = False


@dataclass
class GroupingCfg:
    method: str
    separator: str
    time_interval: int
    grouping_factors: List[int] = field(default_factory=lambda: [0])


@dataclass
class SplittingCfg:
    method: str
    balancing_range: list
    tvt_percentages: list
    train_stems: Optional[List[str]] = None
    val_stems: Optional[List[str]] = None
    test_stems: Optional[List[str]] = None


@dataclass
class SamplingCfg:
    method: str


@dataclass
class ClassifSamplingCfg:
    method: str


@dataclass
class Dataset:
    name: str
    classes: List[str]
    ood: List[str]
    ood_path: Optional[str] = None
    id_path: Optional[str] = None
    img_folder_path: Optional[str] = None
    label_folder_path: Optional[str] = None
    background_img_folder_path: Optional[str] = None
    yaml_path: Optional[str] = None
    sampling_cfg: Optional[SamplingCfg] = None
    grouping_cfg: Optional[GroupingCfg] = None
    splitting_cfg: Optional[SplittingCfg] = None


@dataclass
class Solver:
    criterion: str
    optimizer: str
    scheduler: str
    init_lr: float
    weight_decay: float
    init_lr: float
    max_lr: float
    base_lr: float
    nb_warmup_epoch: int
    nb_decrease_lr_epoch: int
    nb_epochs: int
    weight_decay: float


@dataclass
class TrainConfig:
    target: str
    task: str
    datasets: list[Dataset]
    mc_dropout_iter: Optional[int] = 50
    eval_split: Optional[str] = 'val'
    model: Optional[Model] = None
    augmentation: Optional[dict] = None
    nb_epochs: Optional[int] = 50
    nb_workers: Optional[int] = 4
    img_size: Optional[int] = 640
    batch_size: Optional[int] = 16
    device: Optional[str] = 'cuda'
    solver: Optional[Solver] = None
    sampling: Optional[ClassifSamplingCfg] = None


@dataclass
class EvalConfig:
    target: str
    task: str
    datasets: list[Dataset]
    one_class_eval: Optional[bool] = False
    mc_dropout_iter: Optional[int] = 20
    model: Optional[Model] = None
    nb_workers: Optional[int] = 8
    img_size: Optional[int] = 640
    batch_size: Optional[int] = 16
    device: Optional[str] = 'cuda'
    iou_thresh: Optional[float] = 0.7
    max_detections: Optional[int] = 300
    verbose: Optional[bool] = False
    split: Optional[str] = 'test'
    conf_thresh: Optional[float] = 0.75
    single_class: Optional[bool] = False


@dataclass
class InferConfig:
    target: str
    folders: list
    flower_detection_weights: str
    insect_detection_weights: str
    insect_classification_weights: str


@dataclass
class Config:
    train: Optional[TrainConfig] = None
    eval: Optional[EvalConfig] = None
    infer: Optional[InferConfig] = None


def load_configuration(path_configuration):
    yaml_config = OmegaConf.load(path_configuration)
    configuration = OmegaConf.merge(OmegaConf.structured(Config), yaml_config)
    return configuration
