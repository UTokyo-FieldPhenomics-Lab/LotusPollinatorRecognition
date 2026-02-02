# Standard imports
from pathlib import Path

# Third-party imports
import cv2
import pandas as pd
import torch
import yaml
from collections import Counter
from PIL import Image
from torchvision import transforms
from torch.utils.data import ConcatDataset, Dataset
from ultralytics import YOLO

# Custom imports


class YOLODetectionDataset(Dataset):
    def __init__(self,
                 img_folder_path: Path,
                 label_folder_path: Path,
                 classes,
                 ood_ids,):
        """
        Args:
            img_folder_path (Path): Path to the images folder.
            label_folder_path (Path): Path to the labels folder.
            classes (dict): Dictionary of classes in the dataset.
            ood_classes (dict): Dictionary of classes in the dataset.
        """
        self.img_folder_path = Path(img_folder_path)
        ext = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff', '.webp', '.heic', '.heif', '.avif'}
        self.img_list = [
            p for p in self.img_folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in ext
        ]
        self.img_list.sort()
        self.label_folder_path = Path(label_folder_path)
        self.classes = classes

        # Build label dataframe
        label_paths = list(self.label_folder_path.glob('*.txt'))
        self.label_df = pd.DataFrame()
        for label_path in label_paths:
            img_name = label_path.stem
            if not label_path.exists():
                continue
            with open(label_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    new_row = pd.DataFrame({
                        'img_name': [img_name],
                        'class_id': [class_id],
                        'xc_rel': [x],
                        'yc_rel': [y],
                        'w_rel': [w],
                        'h_rel': [h]
                    })
                    self.label_df = pd.concat([self.label_df, new_row], ignore_index=True)

        # Visualize the dataset distribution
        class_counts = self.label_df['class_id'].value_counts().to_dict()
        class_str = ' '.join([f'{self.classes[class_id]}: {count},' for class_id, count in sorted(class_counts.items())])
        print(class_str)
        print('--------------------------------------------------------')

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.img_list)

    def __getitem__(self, idx):
        """
        Returns an image of the dataset and its associated annotations based on its index.
        """
        img_path = self.img_list[idx]
        img_name = img_path.stem
        label_row = self.label_df[self.label_df['img_name'] == img_name]

        annotations = []  # [id, x, y, w, h]
        for _, row in label_row.iterrows():
            annotations.append([row['class_id'], row['xc_rel'], row['yc_rel'], row['w_rel'], row['h_rel']])

        return img_path, [i[0] for i in annotations], [i[1:] for i in annotations]  # img_path, classes, bounding boxes