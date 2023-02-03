from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Dict, Any

from data.dida_dataset import DidaSegmentationDataset

def get_dataloader(data_dir: str,
                   image_folder: str = 'images',
                   label_folder: str = 'labels',
                   seed: int = 100,
                   fraction: float = 0.2,
                   batch_size: int = 4) -> Dict[str, Any]:
    """Create train and val dataloader from a single directory containing
    the image and label folders.
    Args:
        data_dir (str): Data directory path or root
        image_folder (str, optional): Image folder name. Defaults to 'images'.
        label_folder (str, optional): Label folder name. Defaults to 'labels'.
        fraction (float, optional): Fraction of val set. Defaults to 0.2.
        batch_size (int, optional): Dataloader batch size. Defaults to 4.
    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        train and val dataloaders.
    """
    data_transforms = transforms.Compose([transforms.ToTensor()])

    image_datasets = {
        x: DidaSegmentationDataset(data_dir,
                               image_folder=image_folder,
                               label_folder=label_folder,
                               seed=seed,
                               fraction=fraction,
                               subset=x,
                               transforms=data_transforms)
        for x in ['train', 'val']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=8)
        for x in ['train', 'val']
    }
    return dataloaders
    