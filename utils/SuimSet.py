from pathlib import Path
from typing import Callable


import numpy as np
import torch
import utils.utils as utils
from torch.utils.data import Dataset
from PIL import Image



def download(download_dir: Path):

    test_id="1YZBFO_tEmP5aImfrH01nDu4Kg1wtvklX"
    train_id="1hsTc6XeL59WDQIj0EOMZXdtT-Af0RcOg"

    # loading the train data
    utils.download_from_gdrive_and_extract_zip(
        file_id=train_id,
        save_path=download_dir.joinpath("train_val.zip"),
        extract_path=download_dir.joinpath("train_val/"),
    )

    # loading the test data
    utils.download_from_gdrive_and_extract_zip(
        file_id=test_id,
        save_path=download_dir.joinpath("TEST.zip"),
        extract_path=download_dir.joinpath("TEST/"),
    )

class SuimDataSet(Dataset):
    def __init__(
    self,
    root_path: Path,
    transform_images: Callable = None,
    transform_labels: Callable = None,
    ):
        """
        Initializes the dataset.

        Args:
            root_path (Path): Path to the dataset directory.
            transform_images (callable, optional): Transformation function for images.
            transform_labels (callable, optional): Transformation function for labels.
        """

        self.root_path = root_path
        self.transform_images = transform_images
        self.transform_labels = transform_labels
        self.image_paths = list((root_path / "images").glob("*.jpg"))

        self.rgb_to_class = {
            (0, 0, 0):          0,  # Black - Background waterbody
            (0, 0, 255):        1,  # Blue - Human divers
            (0, 255, 0):        2,  # Green - Plants/sea-grass
            (0, 255, 255):      3,  # Sky blue - Wrecks/ruins
            (255, 0, 0):        4,  # Red - Robots/instruments
            (255, 0, 255):      5,  # Pink - Reefs/invertebrates
            (255, 255, 0):      6,  # Yellow - Fish and vertebrates
            (255, 255, 255):    7,  # White - Sea-floor and rocks
        }


    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor | Image.Image, torch.Tensor, torch.Tensor]:
        """
        Retrieves the image and corresponding label masks for a given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing:
                - image (torch.Tensor | Image.Image): The transformed image or original image.
                - label_masks (torch.Tensor): A binary mask tensor of shape (K, H, W) where K is the number of classes.
                  Each channel represents the binary mask for a specific class.
                - labels_tensor (torch.Tensor): A segmentation map tensor of shape (1, H, W) indicating class indices
                for each pixel.
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        label_path = self.root_path / f"masks/{image_path.stem}.bmp"
        label_img = np.array(Image.open(label_path)) / 255
        labels = np.zeros(label_img.shape[:2], dtype=np.int64)

        for rgb, class_idx in self.rgb_to_class.items():
            x, y, _ = np.where(label_img == np.array(rgb))
            labels[x, y] = class_idx

        ground_truth = torch.tensor(labels).unsqueeze(0)
        label_masks = torch.zeros(len(self.rgb_to_class), *labels.shape).scatter_(0, ground_truth, 1)

        if self.transform_images:
            image = self.transform_images(image)
        if self.transform_labels:
            label_masks = self.transform_labels(label_masks)
            ground_truth = self.transform_labels(ground_truth)

        return image, label_masks, ground_truth