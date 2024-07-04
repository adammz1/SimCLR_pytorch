from PIL import Image
import os
import torch
from torchvision import transforms


class CustomDataset:
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        self.transform = transform

    def __len__(self):
        """Returns the total number of images."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the image to get.

        Returns:
            PIL.Image: Image at the specified index.
        """
        if idx < 0 or idx >= len(self.image_files):
            raise IndexError("Index out of range")

        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name)
        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(0)
