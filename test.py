import os
import torch
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from math import log10, sqrt 

from model import DnCNN
from dataset import DenoisingDataset

def load_and_preprocess(image_path, device):
    """ Load an image, convert to tensor, and send to device """
    image = cv2.imread(image_path)  # Load image (BGR format)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor (scales to [0,1])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image, image_tensor  # Return both original and tensor

def postprocess_and_plot(original, output):
    """ Convert model output back to an image and plot all three images """
    # Convert the output to numpy arrays
    output_image = output.squeeze(0).cpu().detach().numpy()  # Remove batch dim
    output_image = np.clip(output_image, 0, 1)  # Clip values to [0,1]
    output_image = np.transpose(output_image, (1, 2, 0))  # Convert to HWC format

    _, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Original (Noisy) Image
    axes[0].imshow(original)
    axes[0].set_title("Original Image (Noisy)")
    axes[0].axis("off")

    # Denoised Output
    axes[1].imshow(output_image)
    axes[1].set_title("Denoised Output")
    axes[1].axis("off")

    plt.show()


# to measure the level of distortion in the image
def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):
        return 100
    
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 
    

def main():
    # Directories 
    source_dir = os.getcwd()
    train_dir = os.path.join(source_dir, 'data/train')
    test_dir = os.path.join(source_dir, 'data/test/test')
    val_dir = os.path.join(source_dir, 'data/validate/validate')

    classes = ['ground truth', 'noisy images']

    batch_size = 4
    num_workers = 8
    learn_rate = 1e-3
    epochs = 15
    num_layers = 17

    test_dataset = DenoisingDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True, persistent_workers=True)
    
    val_dataset = DenoisingDataset(val_dir)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = DnCNN(channels=3, num_of_layers=17).to(device)  # Initialize model
    model.load_state_dict(torch.load(f"YOUR_MODEL_NAME_HERE.pth"))
    model.eval()

    # Test on an image in the validation dataset
    noisy_image_path = os.path.join(val_dir, classes[1], 'FILENAME.jpg')
    clean_image_path = os.path.join(val_dir, classes[0], 'FILENAME.jpg')

    _, input_tensor = load_and_preprocess(noisy_image_path, device)

    # Forward pass through the model
    with torch.no_grad():
        output_tensor = model(input_tensor)

    clean_image, _ = load_and_preprocess(clean_image_path, device)

    # Postprocess and plot
    postprocess_and_plot(clean_image, output_tensor)


if __name__ == '__main__':
    main()