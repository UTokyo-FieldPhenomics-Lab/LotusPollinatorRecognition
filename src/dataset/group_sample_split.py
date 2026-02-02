# Standard imports
import os
import glob
import random
from pathlib import Path
from collections import defaultdict

# Third-party imports
import clip
import cv2
import torch
import numpy as np
from datetime import datetime
from PIL import Image

# Custom imports


def group_by_time(grouping_cfg, img_paths):

    split_list = [x.split('\\')[-1].split('_') for x in img_paths]

    key_list = [x[0] for x in split_list]

    timestamp_list = [datetime.strptime('_'.join(parts[1:6] + [parts[6].split('.')[0]]), '%Y_%m_%d_%H_%M_%S') for parts in split_list]

    time_interval = grouping_cfg['time_interval']

    bin_key_list = []
    for key, dt in zip(key_list, timestamp_list):
        day_start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        seconds_since_midnight = (dt - day_start).total_seconds()
        bin_key = f"{key}_{dt.year}_{dt.month}_{dt.day}_{int(seconds_since_midnight // time_interval)}"
        bin_key_list.append(bin_key)

    img_splitting_dict = defaultdict(list)
    
    for img_path, bin_key in zip(img_paths, bin_key_list):
        img_splitting_dict[bin_key].append(img_path)

    return dict(img_splitting_dict)


def group_by_consecutive_frame(grouping_cfg, img_paths):

    split_list = [x.split('\\')[-1].split('_') for x in img_paths]

    timestamp_list = [datetime.strptime('_'.join(parts[1:6] + [parts[6].split('.')[0]]), '%Y_%m_%d_%H_%M_%S') for parts in split_list]

    # Sort all lists based on timestamp
    sorted_data = sorted(zip(timestamp_list, img_paths))
    timestamp_list, img_paths = zip(*sorted_data)

    timestamp_list = list(timestamp_list)
    img_paths = list(img_paths)

    img_splitting_dict = defaultdict(list)
    current_group = 0

    if len(timestamp_list) > 0:
        img_splitting_dict[f"group_{current_group}"].append(img_paths[0])

        for i in range(1, len(timestamp_list)):
            time_diff = (timestamp_list[i] - timestamp_list[i - 1]).total_seconds()

            if time_diff <= 9:
                img_splitting_dict[f"group_{current_group}"].append(img_paths[i])
            else:
                current_group += 1
                img_splitting_dict[f"group_{current_group}"].append(img_paths[i])

    return dict(img_splitting_dict)


def group_dataset(grouping_cfg, img_folder_path):

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]
    img_paths = []
    for ext in extensions:
        pattern = os.path.join(img_folder_path, ext)
        img_paths.extend(glob.glob(pattern))

    if grouping_cfg['method'] == 'split':
        img_splitting_dict = defaultdict(list)
        for img_path in img_paths:
            p = Path(img_path)
            parts = p.stem.split(grouping_cfg['separator'])
            key = "".join(parts[i] for i in grouping_cfg['grouping_factors'])
            img_splitting_dict[key].append(str(p.resolve()))

    elif grouping_cfg['method'] == "unique":
        img_splitting_dict = defaultdict(list)
        for img_path in img_paths:
            p = Path(img_path)
            key = p.stem
            img_splitting_dict[key].append(str(p.resolve()))

    elif grouping_cfg['method'] == "time":
        img_splitting_dict = group_by_time(grouping_cfg, img_paths)

    elif grouping_cfg['method'] == "consecutive":
        img_splitting_dict = group_by_consecutive_frame(grouping_cfg, img_paths)

    return dict(img_splitting_dict)


def split_dataset(splitting_cfg, grouped_dict, ood_class_id):

    train_img_list, val_img_list, test_img_list = [], [], []
    val_ood_img_list, test_ood_img_list = [], []

    if splitting_cfg['method'] == 'random':

        train_percentage = splitting_cfg['tvt_percentages'][0]
        val_percentage = splitting_cfg['tvt_percentages'][1]
        test_percentage = splitting_cfg['tvt_percentages'][2]

        for key, img_list in grouped_dict.items():

            # Check if any label contains OOD class
            has_ood = False
            for img_path in img_list:
                label_path = img_path.replace('images', 'labels').replace(os.path.splitext(img_path)[1], '.txt')
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        for line in f:
                            class_id = int(line.split()[0])
                            if class_id in ood_class_id:
                                has_ood = True
                                break
                    if has_ood:
                        break

            rand_val = random.random() * 100

            if has_ood:
                if rand_val < 50:
                    val_ood_img_list.extend(img_list)
                else:
                    test_ood_img_list.extend(img_list)

            else:
                if rand_val < train_percentage:
                    train_img_list.extend(img_list)
                elif rand_val < train_percentage + val_percentage:
                    val_img_list.extend(img_list)
                else:
                    test_img_list.extend(img_list)

        return train_img_list, val_img_list, test_img_list, val_ood_img_list, test_ood_img_list

    elif splitting_cfg['method'] == 'stem':
        for key, img_list in grouped_dict.items():

            first_element = Path(img_list[0]).stem

            if any(stem in first_element for stem in splitting_cfg['train_stems']):
                train_img_list.extend(img_list)

            elif any(stem in first_element for stem in splitting_cfg['val_stems']):
                val_img_list.extend(img_list)

            elif any(stem in first_element for stem in splitting_cfg['test_stems']):
                test_img_list.extend(img_list)

        return train_img_list, val_img_list, test_img_list


def extract_clip_features(crop: np.ndarray, model, preprocess, device) -> torch.Tensor:
    """
    Extract a L2-normalized CLIP image feature vector from a BGR numpy crop.
    """
    if crop is None or crop.size == 0:
        return torch.zeros(model.visual.output_dim)

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    img_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = model.encode_image(img_tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize

    return feats.squeeze(0).detach().cpu()


def sample_from_similarity(label_lst: list):

    # Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    resampled_train_label_lst = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.gif']

    label_path_0 = label_lst[0]
    resampled_train_label_lst.append(label_path_0)

    img_path_0 = label_path_0.replace('labels', 'images').replace('.txt', '')
    for ext in image_extensions:
        if os.path.exists(img_path_0 + ext):
            img_file_0 = img_path_0 + ext
            break

    bbox_list_0 = []
    with open(label_path_0, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            bbox_list_0.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])

    img_0 = cv2.imread(img_file_0)
    h0, w0 = img_0.shape[:2]

    features_0 = []

    for bbox in bbox_list_0:
        cls_id, x, y, w, h = bbox
        x1 = int((x - w / 2) * w0)
        y1 = int((y - h / 2) * h0)
        x2 = int((x + w / 2) * w0)
        y2 = int((y + h / 2) * h0)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w0, x2), min(h0, y2)

        crop = img_0[y1:y2, x1:x2]
        crop = cv2.resize(crop, (224, 224))

        feature = extract_clip_features(crop, model, preprocess, device)
        features_0.append(feature)

    for label_path_1 in label_lst[1:]:

        bbox_list_1 = []

        img_path_1 = label_path_1.replace('labels', 'images').replace('.txt', '')

        for ext in image_extensions:
            if os.path.exists(img_path_1 + ext):
                img_file_1 = img_path_1 + ext
                break

        with open(label_path_1, 'r') as f:
            lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                bbox_list_1.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])

        img_1 = cv2.imread(img_file_1)

        h1, w1 = img_1.shape[:2]

        features_1 = []

        for bbox in bbox_list_1:
            cls_id, x, y, w, h = bbox
            x1 = int((x - w/2) * w1)
            y1 = int((y - h/2) * h1)
            x2 = int((x + w/2) * w1)
            y2 = int((y + h/2) * h1)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w1, x2), min(h1, y2)

            crop = img_1[y1:y2, x1:x2]
            crop = cv2.resize(crop, (224, 224))

            feature = extract_clip_features(crop, model, preprocess, device)
            features_1.append(feature)

        if len(features_0) > 0 and len(features_1) > 0:
            F0 = torch.stack(features_0, dim=0)  # (N0, D)
            F1 = torch.stack(features_1, dim=0)  # (N1, D)
            similarity_matrix = F0 @ F1.T  # cosine similarity

            if np.mean(similarity_matrix.numpy()) < 0.98:
                resampled_train_label_lst.append(label_path_1)
                print(f"{label_path_1} included.")

        features_0 = features_1

    return resampled_train_label_lst
                

def sample_dataset(sampling_cfg: dict, img_list: list):

    if sampling_cfg['method'] == 'no':
        return img_list

    elif sampling_cfg['method'] == 'similarity':
        return sample_from_similarity(img_list)
