# Standard imports

# Third-party imports
from ultralytics import YOLO

# Custom imports
from .feature_extractor import FeatureExtractor, get_embedding


def build_detector(architecture, checkpoint):

    if architecture == 'yolo':
        model = YOLO(checkpoint)

    return model
