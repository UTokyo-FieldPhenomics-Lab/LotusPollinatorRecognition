# Standard imports
from pathlib import Path
import matplotlib.pyplot as plt

# Third-party imports
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from torchvision.models import resnet18
from tqdm import tqdm
from torchvision import transforms

# Custom imports


GLOBAL_SEED = 42


def infer_function(infer_configuration, run_name):
    flower_detection_model = YOLO(infer_configuration.flower_detection_weights).cuda()

    insect_detection_model = YOLO(infer_configuration.insect_detection_weights).cuda()

    flower_output_dir = Path('runs/infer') / run_name / 'flower'
    insect_output_dir = Path('runs/infer') / run_name / 'insect'

    flower_output_dir.mkdir(parents=True, exist_ok=True)
    insect_output_dir.mkdir(parents=True, exist_ok=True)

    flower_list = []
    insect_list = []

    for folder in infer_configuration.folders[::-1]:

        image_paths = list(Path(folder).rglob('*.jpg')) + list(Path(folder).rglob('*.png'))

        for image_path in tqdm(image_paths, desc=f'Processing {Path(folder).name}'):

            flower_results = flower_detection_model(image_path)

            if not flower_results or not flower_results[0].boxes:
                continue

            original_image = Image.open(image_path)
            r = flower_results[0]

            boxes = r.boxes
            areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
            largest_box_index = torch.argmax(areas)
            largest_box = boxes[largest_box_index]
            flower_bbox = largest_box.xyxy[0].cpu().numpy()
            confidence = largest_box.conf[0].item()
            flower_list.append({'img_path': str(image_path),
                                'bbox': flower_bbox.tolist(),
                                'confidence': confidence,})
            confidence = largest_box.conf[0].item()

            # image_with_flower_box = original_image.copy()
            # draw = ImageDraw.Draw(image_with_flower_box)
            # draw.rectangle(flower_bbox.tolist(), outline='white', width=5)
            # font = ImageFont.truetype("arial.ttf", 100)
            # text = f"Lotus Flower {confidence:.2f}"
            # text_position = (flower_bbox[0], flower_bbox[1] - 200)
            # draw.text(text_position, text, fill='white', font=font)
            # save_path_flower = flower_output_dir / f'{image_path.stem}_largest_flower.jpg'
            # image_with_flower_box.save(save_path_flower)

            cropped_flower = original_image.crop(flower_bbox.tolist())

            insect_results = insect_detection_model(cropped_flower)

            # image_with_insect_boxes = cropped_flower.copy()
            # draw_insect = ImageDraw.Draw(image_with_insect_boxes)

            for insect_r in insect_results:
                if not insect_r.boxes:
                    continue

                for insect_box in insect_r.boxes:
                    insect_bbox = insect_box.xyxy[0].cpu().numpy()
                    class_id = int(insect_box.cls)
                    class_name = insect_detection_model.names[class_id]
                    insect_confidence = insect_box.conf[0].item()

                    insect_list.append({'img_path': str(image_path),
                                        'bbox': insect_bbox.tolist(),
                                        'class_id': class_id,
                                        'confidence': insect_confidence})

                    # draw_insect.rectangle(insect_bbox.tolist(), outline='blue', width=3)
                    # font_insect = ImageFont.truetype("arial.ttf", 15)
                    # draw_insect.text((insect_bbox[0], insect_bbox[1] - 15), class_name, fill='blue', font=font_insect)

            # save_path_insect = insect_output_dir / f'{image_path.stem}_largest_flower_insects.jpg'
            # image_with_insect_boxes.save(save_path_insect)

        pd.DataFrame(flower_list).to_csv(f'runs/infer/{run_name}/{Path(folder).stem}_flower.csv', index=False)
        pd.DataFrame(insect_list).to_csv(f'runs/infer/{run_name}/{Path(folder).stem}_insect.csv', index=False)