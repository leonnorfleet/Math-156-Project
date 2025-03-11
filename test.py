import os
import torch
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms
import numpy as np

from model import DnCNN

def load_and_preprocess(image_path, device):
    """ Load an image, convert to tensor, and send to device """
    image = cv2.imread(image_path)  # Load image (BGR format)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor (scales to [0,1])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    return image, image_tensor  # Return both original and tensor

def postprocess_and_plot(original, output):
    """ Convert model output back to an image and plot all three images """
    # Convert the output to numpy arrays
    output_image = output.squeeze(0).cpu().detach().numpy()  # Remove batch dim
    output_image = np.clip(output_image, 0, 1)  # Clip values to [0,1]
    output_image = np.transpose(output_image, (1, 2, 0))  # Convert to HWC format
    
    noisy_image = original / 255.0  # Convert noisy image from [0,255] to [0,1]

    # Calculate the difference (original - output)
    difference_image = noisy_image - output_image

    # Plot all three images: original, output, and the difference
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Original (Noisy) Image
    axes[0].imshow(original)
    axes[0].set_title("Original Image (Noisy)")
    axes[0].axis("off")

    # Denoised Output
    axes[1].imshow(output_image)
    axes[1].set_title("Denoised Output")
    axes[1].axis("off")

    # Difference Image (Original - Output)
    # axes[2].imshow(difference_image)
    # axes[2].set_title("Difference (Original - Output)")
    # axes[2].axis("off")

    plt.show()

def main():
    source_dir = os.getcwd()
    train_dir = os.path.join(source_dir, 'data/train')
    test_dir = os.path.join(source_dir, 'data/test')
    val_dir = os.path.join(source_dir, 'data/validate')
    classes = ['ground truth', 'noisy images']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_path = os.path.join(val_dir, classes[1], '45611010422_08bb2df799_c.jpg')

    model = DnCNN(channels=3, num_of_layers=14).to(device)  # Initialize model
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    original_image, input_tensor = load_and_preprocess(image_path, device)

    # Forward pass through the model
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Postprocess and plot
    postprocess_and_plot(original_image, output_tensor)


if __name__ == '__main__':
    main()