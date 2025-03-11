import os
import shutil

source_dir = os.getcwd()
source_dir = os.path.join(source_dir, 'data')
folders = ['test', 'train', 'validate']

noises = ['gauss_', 'poisson_', 'salt and pepper_', 'speckle_']

for name in folders:
    ni = os.path.join(source_dir, name, name, 'noisy images')
    # os.makedirs(target_dir, exist_ok=True)

    for filename in os.listdir(ni):
        for noise in noises:
            if noise in filename:
                old = os.path.join(ni, filename)
                new = os.path.join(ni, filename.removeprefix(noise))
                os.rename(old, new)
                    