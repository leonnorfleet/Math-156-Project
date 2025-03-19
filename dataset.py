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
        clean_path = os.path.join(self.clean_dir, self.filenames[idx])
        
        # Load images and ensure consistent format
        noisy_image = io.read_image(noisy_path, mode=io.ImageReadMode.RGB)  # Force RGB mode
        clean_image = io.read_image(clean_path, mode=io.ImageReadMode.RGB)  # Force RGB mode
        
        # Convert to float32 explicitly and normalize
        noisy_image = noisy_image.float().div(255.0)
        clean_image = clean_image.float().div(255.0)
        
        return noisy_image, clean_image
