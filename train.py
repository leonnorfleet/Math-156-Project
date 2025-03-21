'''
image enhancement CNN using DnCNN

'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import time

from model import DnCNN
from dataset import DenoisingDataset


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Directories 
    source_dir = os.getcwd()
    train_dir = os.path.join(source_dir, 'data/train')

    transform = transforms.ToTensor()
    batch_size = 4
    num_workers = 8
    learn_rate = 1e-3
    epochs = 15
    num_layers = 14

    classes = ['ground truth', 'noisy images']

    # Create Datasets
    train_dataset = DenoisingDataset(train_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)

    model = DnCNN(channels=3, num_of_layers=num_layers).to(device)

    torch.backends.cudnn.benchmark = True

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    scaler = torch.GradScaler()

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_begin = time.time()

        for i, data in enumerate(train_loader, 0):
            noisy_image, clean_image = data

            noisy_image = noisy_image.to(device, dtype=torch.float16)
            clean_image = clean_image.to(device, dtype=torch.float16)
            
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                output = model(noisy_image)
                loss = criterion(output, clean_image).div_(noisy_image.size(0) * 2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        torch.cuda.empty_cache()

        epoch_end = time.time() - epoch_begin
        print(f'Time taken for 1 epoch: {epoch_end:.2f} seconds')

    end_time = time.time() - start_time
    print(f'Total time taken: {end_time:.2f} seconds')
    torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()