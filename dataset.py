import os
import torch
from torch.utils.data import Dataset
import torchvision.io as io

class DenoisingDataset(Dataset):
    def __init__(self, root_dir):
        self.noisy_dir = os.path.join(root_dir, "noisy images")
        self.clean_dir = os.path.join(root_dir, "ground truth")
        self.filenames = sorted(os.listdir(self.noisy_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.filenames[idx])
        clean_path = os.path.join(self.clean_dir, self.filenames[idx])  # Assumes same name

        # Load images as tensors without altering format
        noisy_image = io.read_image(noisy_path)  # Loads as (C, H, W) tensor
        clean_image = io.read_image(clean_path)  # Loads as (C, H, W) tensor

        return noisy_image.float() / 255.0, clean_image.float() / 255.0  # Normalize if needed
