# Standard imports
import glob
import os
from collections import defaultdict
from pathlib import Path

# Third-party imports
import yaml

# Custom imports
from src.model import build_detector
from src.dataset import group_dataset, split_dataset, sample_dataset
from src.utils import create_symlinks


GLOBAL_SEED = 42


def print_class_distribution(img_list, set):
    class_counts = defaultdict(int)

    for img_path in img_list:
        label_path = str(Path(img_path).with_suffix(".txt")).replace("images", "labels")

        if not Path(label_path).exists():
            continue
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                class_counts[class_id] += 1

    sorted_counts = dict(sorted(class_counts.items()))
    print(f"Class distribution in the {set} set: {sorted_counts}")


def train_recognition(run_name,
                   batch_size: int,
                   datasets_configuration,
                   detector_configuration,
                   device: str,
                   img_size: int,
                   nb_epochs: int,
                   nb_workers: int):

    run_path = Path("runs/train") / run_name

    model = build_detector(detector_configuration.architecture, detector_configuration.checkpoint)

    datasets = []

    for dataset in datasets_configuration:
        class_list = dataset.classes
        ood_list = dataset.ood

        # Cluster images in small groups based on the chosen rule
        grouped_dict = group_dataset(dataset.grouping_cfg, dataset.img_folder_path)

        # Split image groups in train, test and val sets
        ood_class_id = [i for i, c in enumerate(class_list) if c in ood_list]
        train_img_list, val_img_list, test_img_list, val_ood_img_list, test_ood_img_list = split_dataset(dataset.splitting_cfg, grouped_dict, ood_class_id)

        # Resample unbalanced train sets
        train_img_list = sample_dataset(dataset.sampling_cfg, train_img_list)

        # Deal with background images
        # ext = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff', '.webp', '.heic', '.heif', '.avif'}
        # background_img_paths = sorted([p for p in Path(dataset.background_img_folder_path).iterdir() if p.is_file() and p.suffix.lower() in ext],key=lambda x: x.name)
        # train_pct, val_pct, _ = dataset.splitting_cfg.tvt_percentages
        # total_bg = len(background_img_paths)
        # train_bg_count, val_bg_count = int(total_bg * train_pct / 100), int(total_bg * val_pct / 100)
        # train_img_list.extend(background_img_paths[:train_bg_count])
        # val_img_list.extend(background_img_paths[train_bg_count:train_bg_count + val_bg_count])
        # test_img_list.extend(background_img_paths[train_bg_count + val_bg_count:])

        # Redefine classes with ood
        in_dist_classes = sorted([c for c in class_list if c not in ood_list])
        filtered_classes = in_dist_classes + sorted(ood_list)
        class_remapping = {original_id: filtered_classes.index(class_name) for original_id, class_name in enumerate(class_list)}

        # Fix cropped datasets
        train_img_list = glob.glob(os.path.join(
            r'C:\Users\CREST\Documents\GitHub\insect-recognition\runs\train\original_insect_no_background-detection-251109-122524-rare-cod\dataset\train\images',
            '**', '*.jpg'), recursive=True)
        test_img_list = glob.glob(os.path.join(
            r'C:\Users\CREST\Documents\GitHub\insect-recognition\runs\train\original_insect_no_background-detection-251109-122524-rare-cod\dataset\test\images',
            '**', '*.jpg'), recursive=True)
        val_img_list = glob.glob(os.path.join(
            r'C:\Users\CREST\Documents\GitHub\insect-recognition\runs\train\original_insect_no_background-detection-251109-122524-rare-cod\dataset\val\images',
            '**', '*.jpg'), recursive=True)
        # test_ood_img_list = glob.glob(os.path.join(
        #     r'C:\Users\CREST\Documents\GitHub\insect-recognition\runs\train\cropped_insect-detection-251106-235057-alive-ghost\dataset\test_ood\images',
        #     '**', '*.jpg'), recursive=True)
        # val_ood_img_list = glob.glob(os.path.join(
        #     r'C:\Users\CREST\Documents\GitHub\insect-recognition\runs\train\cropped_insect-detection-251106-235057-alive-ghost\dataset\val_ood\images',
        #     '**', '*.jpg'), recursive=True)

        # train_img_list = [str(x) for x in train_img_list if '1_cropped_no_insect' not in str(x)]
        # val_img_list = [str(x) for x in val_img_list if '1_cropped_no_insect' not in str(x)]
        # test_ood_img_list = [str(x) for x in test_ood_img_list if '1_cropped_no_insect' not in str(x)]
        # train_img_list = [path_str for path_str in train_img_list if '1_cropped_no_insect' not in str(Path(path_str).resolve())]
        # train_img_list = [x.replace('1_yolo_cropped', '1_yolo_original') for x in
        #                   [str(Path(p).resolve()) for p in train_img_list if Path(p).is_symlink()]]
        # train_img_list = [y for y in [x.replace('1_yolo_cropped', '1_yolo_original')
        #                               for x in [str(Path(p).resolve()) for p in train_img_list if Path(p).is_symlink()]]
        #                   if '1_cropped_no_insect' not in y]
        # train_img_list = [y.replace('1_cropped_no_insect', '1_original_no_insect') for y in
        #                  [x.replace('1_yolo_cropped', '1_yolo_original')
        #                   for x in [str(Path(p).resolve()) for p in train_img_list if Path(p).is_symlink()]]]
        # test_img_list = [y.replace('1_cropped_no_insect', '1_original_no_insect') for y in
        #                  [x.replace('1_yolo_cropped', '1_yolo_original')
        #                               for x in [str(Path(p).resolve()) for p in test_img_list if Path(p).is_symlink()]]]
        # val_img_list = [y.replace('1_cropped_no_insect', '1_original_no_insect') for y in
        #                  [x.replace('1_yolo_cropped', '1_yolo_original')
        #                   for x in [str(Path(p).resolve()) for p in val_img_list if Path(p).is_symlink()]]]
        # test_ood_img_list = [x.replace('1_yolo_cropped', '1_yolo_original')
        #                               for x in [str(Path(p).resolve()) for p in test_ood_img_list if Path(p).is_symlink()]]
        # val_ood_img_list = [x.replace('1_yolo_cropped', '1_yolo_original')
        #                               for x in [str(Path(p).resolve()) for p in val_ood_img_list if Path(p).is_symlink()]]
        # test_img_list = [x.replace('1_yolo_cropped', '1_yolo_original') for x in
        #                   [str(Path(p).resolve()) for p in test_img_list if Path(p).is_symlink()]]
        # val_img_list = [x.replace('1_yolo_cropped', '1_yolo_original') for x in
        #                  [str(Path(p).resolve()) for p in val_img_list if Path(p).is_symlink()]]
        class_remapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

        # Create train, test, val sets with symlinks (lightweight files)
        create_symlinks(train_img_list, run_path / "dataset" / "train", class_remapping)
        create_symlinks(val_img_list, run_path / "dataset" / "val", class_remapping)
        create_symlinks(test_img_list, run_path / "dataset" / "test", class_remapping)
        create_symlinks(val_ood_img_list, run_path / "dataset" / "val_ood", class_remapping)
        create_symlinks(test_ood_img_list, run_path / "dataset" / "test_ood", class_remapping)
        print_class_distribution(train_img_list, 'train')
        print_class_distribution(val_img_list, 'val')
        print_class_distribution(test_img_list, 'test')

    data_yaml = {
        "train": str(Path.cwd() / run_path / "dataset" / "train" / "images"),
        "val": str(Path.cwd() / run_path / "dataset" / "val" / "images"),
        "test": str(Path.cwd() / run_path / "dataset" / "test" / "images"),
        "names": in_dist_classes,
        "nc": len(in_dist_classes),
    }

    yaml_path = run_path / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f)

    # training_results =  model.train(batch=batch_size,
    #                                 data=yaml_path,
    #                                 workers=nb_workers,
    #                                 single_cls=False,
    #                                 epochs=nb_epochs,
    #                                 imgsz=img_size,
    #                                 project=run_path,
    #                                 name="results",
    #                                 device=device)

    model.train(
        data=yaml_path,
        workers=nb_workers,
        single_cls=False,
        epochs=nb_epochs,
        imgsz=img_size,
        project=run_path,
        name="results",
        device=device,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        translate=0.0,
        scale=0.0,
        fliplr=0.0,
        mosaic=0.0,
        erasing=0.0,
        auto_augment=None,
    )


def train_function(train_configuration, run_name):

    train_recognition(run_name,
                       train_configuration.batch_size,
                       train_configuration.datasets,
                       train_configuration.model,
                       train_configuration.device,
                       train_configuration.img_size,
                       train_configuration.nb_epochs,
                       train_configuration.nb_workers)
