# Standard imports
from pathlib import Path
import torch

# Third-party imports
import pandas as pd

# Custom imports
from src.model import build_detector, get_embedding
from src.dataset import YOLODetectionDataset
from src.evaluation import compute_ood_metrics, get_matchings
from src.utils import visualize_fifty_one, visualize_embedding_cluster


GLOBAL_SEED = 42


def eval_recognition(run_name, batch_size, conf_thresh, dataset_configuration, device, img_size, iou_thresh, max_detections,
                     model, nb_workers):

    model = build_detector(model.architecture, model.checkpoint)

    metrics = model.val(data=dataset_configuration.yaml_path,
                        iou=iou_thresh,
                        conf=conf_thresh,
                        device=device,
                        imgsz=img_size,
                        max_det=max_detections,
                        split='test',
                        batch=batch_size,
                        workers=nb_workers,
                        visualize=True,
                        save_json=True,
                        save_txt=True,
                        project=Path(r'C:\Users\CREST\Documents\GitHub\insect-recognition\runs\eval') / run_name)

    print(metrics.results_dict)


def eval_ood(run_name, batch_size, conf_thresh, dataset_cfg, device, img_size, iou_thresh, max_detections, model, nb_workers):

    # Redefine classes with ood
    filtered_classes = sorted([c for c in dataset_cfg.classes if c not in dataset_cfg.ood]) + sorted(dataset_cfg.ood)
    class_remapping = {original_id: filtered_classes.index(class_name) for original_id, class_name in enumerate(dataset_cfg.classes)}
    ood_id_list = [filtered_classes.index(c) for c in dataset_cfg.ood]

    # Load model
    model = build_detector(model.architecture, model.checkpoint)

    # Load datasets
    print('Loading in-distribution dataset...')
    id_dataset = YOLODetectionDataset(Path(dataset_cfg.id_path)/'images',
                                   Path(dataset_cfg.id_path)/'labels',
                                   filtered_classes,
                                   ood_id_list)

    print('Loading out-of-distribution dataset...')
    ood_dataset = YOLODetectionDataset(Path(dataset_cfg.ood_path) / 'images',
                                   Path(dataset_cfg.ood_path) / 'labels',
                                   filtered_classes,
                                   ood_id_list)

    ood_df = pd.DataFrame(get_matchings(batch_size, conf_thresh, ood_dataset, device, img_size, iou_thresh, max_detections, model))
    # ood_df = get_embedding(ood_df, run_name)

    # id_df = pd.DataFrame(get_matchings(batch_size, conf_thresh, id_dataset, device, img_size, iou_thresh, max_detections, model))
    # id_df = get_embedding(id_df, run_name)

    # visualize_embedding_cluster(id_df, ood_df, filtered_classes)

    # visualize_fifty_one(id_df, ood_df, filtered_classes)

    compute_ood_metrics(ood_df, filtered_classes)

    # viz_size_graph(ood_df, filtered_classes)


def eval_function(eval_configuration, run_name):

    if eval_configuration.datasets[0].ood_path:
        eval_ood(run_name,
                 eval_configuration.batch_size,
                 eval_configuration.conf_thresh,
                 eval_configuration.datasets[0],
                 eval_configuration.device,
                 eval_configuration.img_size,
                 eval_configuration.iou_thresh,
                 eval_configuration.max_detections,
                 eval_configuration.model,
                 eval_configuration.nb_workers)

    else:
        eval_recognition(run_name,
                         eval_configuration.batch_size,
                         eval_configuration.conf_thresh,
                         eval_configuration.datasets[0],
                         eval_configuration.device,
                         eval_configuration.img_size,
                         eval_configuration.iou_thresh,
                         eval_configuration.max_detections,
                         eval_configuration.model,
                         eval_configuration.nb_workers)
