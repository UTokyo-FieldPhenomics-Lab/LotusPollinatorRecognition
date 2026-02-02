# Standard imports
import os
from pathlib import Path

# Third-party imports
import clip
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from transformers import pipeline

# Custom imports


class FeatureExtractor:
    def __init__(self,
                 model_architecture: str,
                 extraction_stage: str = 'final',
                 device: str = "cuda"):
        """

        Args:
            model_architecture (str): Architecture of the model to be used for feature extraction.
            extraction_stage (str): Stage of the model to be used for feature extraction.
        """
        self.model_architecture = model_architecture
        self.extraction_stage = extraction_stage
        self.device = device

        if model_architecture in ["ViT-B/32",]:
            if self.extraction_stage == "final":
                self.model, self.preprocess = clip.load(model_architecture, device=device)
                self.model.eval()

        if model_architecture in ["DinoV3",]:
            if self.extraction_stage == "final":
                self.model = pipeline(model="facebook/dinov3-vitb16-pretrain-lvd1689m", task="image-feature-extraction", token='')

    def get_features_from_imgpath(self, image_path: str):
        """
        Returns features extracted from an image.
        """

        if self.model_architecture in ["ViT-B/32", ]:

            pil_img = Image.open(image_path)
            pil_img = pil_img.convert("RGB")
            tensor_img = self.preprocess(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model.encode_image(tensor_img)
                features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize

            return features.squeeze(0).detach().cpu()

        elif self.model_architecture in ["DinoV3", ]:

            pil_img = Image.open(image_path)
            pil_img = pil_img.convert("RGB")

            tokens = self.model(pil_img)[0]

            class_token = tokens[:1]
            patch_tokens = tokens[5:]

            class_token_array = np.array(class_token)

            patch_token_array = np.array(patch_tokens)
            patch_token_array = patch_token_array.reshape(14, 14, 768)

            return class_token_array, patch_token_array

    def get_features_from_pilimg(self, pil_img: Image):
        """
        Returns features extracted from an image.
        """

        if self.model_architecture in ["ViT-B/32", ]:

            return 0

        if self.model_architecture in ["DinoV3", ]:

            tokens = self.model(pil_img)[0]

            class_token = tokens[:1]
            patch_tokens = tokens[5:]

            class_token_array = np.array(class_token)

            patch_token_array = np.array(patch_tokens)
            patch_token_array = patch_token_array.reshape(14, 14, 768)

            return class_token_array, patch_token_array


def get_embedding(df, run_name):
    """
    Extract embeddings for images in DataFrame.
    """

    feature_extractor = FeatureExtractor('DinoV3')

    embedding_list = []
    new_path = []

    cropped_img_path = Path(r'C:\Users\CREST\Documents\GitHub\insect-recognition\runs\eval') / run_name / 'crops'
    os.makedirs(cropped_img_path, exist_ok=True)

    df = df[df['match_type']=='matched']



    for idx, row in df.iterrows():
        img_path = cropped_img_path / f"{Path(row['img_path']).stem}_{int(row['gt_id'])}_{idx}.jpg"
        if row['pred_bbox']:
            img = Image.open(row['img_path'])
            img_width, img_height = img.size
            x1, y1, x2, y2 = [coord * img_width if i % 2 == 0 else coord * img_height for i, coord in enumerate(row['pred_bbox'])]
            df.at[idx, 'x1'] = x1
            df.at[idx, 'y1'] = y1
            df.at[idx, 'x2'] = x2
            df.at[idx, 'y2'] = y2
            cropped_img = img.crop((x1, y1, x2, y2)).convert("RGB")
            cropped_img.save(img_path)
            # plt.figure(figsize=(8, 8))
            # plt.imshow(cropped_img)
            # plt.axis('off')
            # plt.show()
            class_token, patch_token = feature_extractor.get_features_from_pilimg(cropped_img)
            embedding_list.append(class_token[0])
            new_path.append(img_path)

    df['embedding'] = embedding_list
    df['new_img_path'] = new_path

    return df
