from typing import Optional, Callable, Any, Tuple
from pathlib import Path
import numpy as np
from PIL import Image

from torchvision.datasets.vision import VisionDataset

import utils


class RoofTopSegmentationDataset(VisionDataset):
    """Custom dataset class for rooftop image segmentation
    """
    def __init__(self,
                  root: Path,
                  image_folder: str,
                  label_folder: str,
                  seed: int,
                  transforms: Optional[Callable] = None,
                  fraction: float = None,
                  subset: str = None,
                ) -> None:

        super().__init__(root, transforms)

        image_folder_path = self.root / image_folder
        label_folder_path = self.root / label_folder
        self.seed = seed

        if not image_folder_path.exists():
            raise OSError(f"{image_folder_path} does not exist.")
        if not label_folder_path.exists():
            raise OSError(f"{label_folder_path} does not exist.")

        if not fraction:
            self.image_names = self._get_sorted_filenames(image_folder_path)
            self.label_names = self._get_sorted_filenames(label_folder_path)
        else:
            if subset.lower() not in ["train", "val"]:
                raise (ValueError(
                       f"{subset} is not a valid input. Acceptable values are train and val."
                      ))
            self.image_list = self._get_sorted_filenames(image_folder_path)
            self.label_list = self._get_sorted_filenames(label_folder_path)

            shuffled_indices = self._get_shuffled_indices(len(self.image_list))
            self.image_list = self.image_list[shuffled_indices]
            self.label_list = self.label_list[shuffled_indices]

            self.fraction = fraction
            split_idx = int(np.ceil(len(self.image_list) * (1 - self.fraction)))

            if subset.lower() == "train":
                self.image_names, self.label_names = self._get_train(split_idx)
            else:
                self.image_names, self.label_names = self._get_val(split_idx)

    # VisionDataset abstract functions defined
    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        image_path = self.image_names[index]
        label_path = self.label_names[index]

        with open(image_path, "rb") as image_file, open(label_path,
                                                      "rb") as label_file:
            image = Image.open(image_file)

            if image.mode != 'RGB':
                image = image.convert('RGB')

            label = Image.open(label_file)

            sample = {"image": image, "label": label}
            if self.transforms:
                utils.set_seed(self.seed)
                sample["image"] = self.transforms(sample["image"])
                utils.set_seed(self.seed)
                sample["label"] = self.transforms(sample["label"])
                sample["label"][sample["label"] > 0] = 1

        return sample

    # helper functions within the class
    def _get_train(self, index: int) -> Tuple:
        """Get arrays of images and labels paths for training
        Args:
            index: index until where training data paths is extracted from all data paths arrays
        Returns:
            tuple of image and label path arrays
        """
        return self.image_list[:index], self.label_list[:index]

    def _get_val(self, index: int) -> Tuple:
        """Get arrays of images and labels paths for validation
        Args:
            index: index from where validation data paths is extracted from all data paths arrays
        Returns:
            tuple of image and label path arrays
        """
        return self.image_list[index:], self.label_list[index:]

    def _get_sorted_filenames(self, folder_path: Path) -> np.ndarray:
        """Get all the filenames in a folder and sort them
        Args:
            folder_path: path from where filenames are to be extracted
        Returns:
            array of all the filenames in the folder
        """
        return np.array(sorted(list(folder_path.glob("*.png"))))

    def _get_shuffled_indices(self, length: int) -> np.ndarray:
        """Shuffle indices so as to pseudo-randomize the train and val splitting
        Args:
            length: number of samples in the dataset for shuffling
        Returns:
            array of shuffled indices
        """
        np.random.seed(self.seed)
        indices = np.arange(length)
        np.random.shuffle(indices)
        return indices
    