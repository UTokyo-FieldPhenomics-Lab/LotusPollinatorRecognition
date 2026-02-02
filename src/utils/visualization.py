# Standard imports
import os
from pathlib import Path

# Third-party imports
import cv2
# import fiftyone as fo
# import fiftyone.brain as fob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Custom imports
from .bbox_operations import xcycwh_2_x1y1x2y2


def visualize_eval_result_one_image(image_path, save_path, gt_bboxes, pred_bboxes):
    """
    Visualize evaluation results for a single image with ground truth and predicted bounding boxes.

    Args:
        image_path (Path): Path to the image.
        save_path (Path): Path to save the visualization.
        gt_bboxes (list): List of ground truth bounding boxes in [xc, yc, w, h] format (normalized).
        pred_bboxes (list): List of predicted bounding boxes in [xc, yc, w, h] format (normalized).

    """
    # Load the image
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]

    # Create the figure
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)

    # Add gt bboxes
    for bbox in gt_bboxes:
        x1, y1, x2, y2 = xcycwh_2_x1y1x2y2(bbox[:4])
        x1, y1, x2, y2 = x1*width, y1*height, x2*width, y2*height
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

    # Add predicted bboxes
    for bbox in pred_bboxes:
        x1, y1, x2, y2 = xcycwh_2_x1y1x2y2(bbox[:4])
        x1, y1, x2, y2 = x1*width, y1*height, x2*width, y2*height
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    # Save figure
    blue_patch = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='blue', linewidth=2)
    red_patch = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='red', linewidth=2)
    ax.legend([blue_patch, red_patch], ['Ground Truth', 'Prediction'], loc='upper right')
    ax.axis('off')
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def visualize_fifty_one(df1, df2, class_list):
    """
    Visualize dataset using FiftyOne with UMAP embeddings.
    """
    df = pd.concat([df1, df2], ignore_index=True)

    dataset = fo.Dataset()

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing visible insects"):
        sample = fo.Sample(filepath=row['new_img_path'])
        sample['class_name'] = class_list[row['gt_class']]
        sample['embeddings'] = np.asarray(row['embedding']).reshape(-1).astype(float).tolist()
        sample['ground_truth'] = fo.Detections(detections=[
            fo.Detection(
                label=class_list[row['gt_class']],
                bounding_box=[
                    row['x1'], row['y1'],
                    row['x2'] - row['x1'],
                    row['y2'] - row['y1']
                ]
            )
        ])
        dataset.add_sample(sample)

    fob.compute_visualization(dataset, embeddings='embeddings', method='umap', brain_key='umap')

    session = fo.launch_app(dataset)
    session.wait()


def visualize_embedding_cluster(df1, df2, class_list):
    """
    Clusters embeddings and visualizes with colors being gt_class column.

    Arguments:
        df1 (pd.DataFrame): DataFrame with columns 'gt_class', 'embedding'.
        df2 (pd.DataFrame): DataFrame with columns 'gt_class', 'embedding'.
        class_list (list): List of class names.
    """
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE

    df = pd.concat([df1, df2], ignore_index=True)

    embeddings = np.array(df['embedding'].tolist())

    # Cluster embeddings using KMeans
    n_clusters = len(class_list)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(embeddings)

    # Reduce dimensionality using t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df) - 1))
    embeddings_2d = tsne.fit_transform(embeddings)
    df['tsne1'] = embeddings_2d[:, 0]
    df['tsne2'] = embeddings_2d[:, 1]

    # Create scatter plot
    plt.figure(figsize=(12, 10))
    unique_classes = sorted(df['gt_class'].unique())
    colors = plt.cm.get_cmap('viridis', len(unique_classes))

    for i, class_id in enumerate(unique_classes):
        class_data = df[df['gt_class'] == class_id]
        plt.scatter(class_data['tsne1'], class_data['tsne2'], color=colors(i), label=class_list[class_id])

    plt.title('t-SNE Visualization of Embeddings by Ground Truth Class')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title="Classes")
    plt.grid(True)
    plt.show()

