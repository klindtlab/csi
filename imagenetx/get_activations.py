import os
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
import matplotlib.pyplot as plt
import torch
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
import os
from imagenet_x import load_annotations


SAVE_PATH = 'MY/results'  # Ensure this folder exists
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

data_dir = 'PATH/TO/imagenet/val/'

# Check if a GPU is available and use it; otherwise, fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = load_annotations()


files = os.listdir(data_dir)

# get overlap in files due to ImageNet version mismatch
len(set(df['file_name']).intersection(set(os.listdir(data_dir))))
samples = []
for i in range(len(df)):
    if df.iloc[i]['file_name'] in files:
        samples.append(
            (data_dir + '%s' % df.iloc[i]['file_name'],
             df.iloc[i]['class'],
             df.iloc[i]
             )
        )

class CustomImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        Args:
            image_paths (list): List of paths to image files.
            targets (list): List of target values (e.g., labels).
            transform (callable, optional): Optional transform to be applied
                on an image.
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx][0]
        image = Image.open(img_path).convert("RGB")  # Load image and convert to RGB
        target = self.samples[idx][2].values[1:-3].astype(int)

        if self.transform:
            image = self.transform(image)

        return image, target
    

def process_and_save(model, dataloader, file, save_every_n=1000):
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Move model to the selected device (GPU or CPU)

    all_outputs = []
    all_targets = []

    with torch.no_grad():  # Disable gradient calculation
        for i, (inputs, targets) in enumerate(tqdm(dataloader, desc="Processing")):
            # Move inputs and targets to the device (GPU or CPU)
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass through the model
            outputs = model(inputs.reshape(inputs.shape[0], -1))

            # Move the data back to CPU and convert it to NumPy for saving
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            # Save intermittently to avoid memory issues or data loss
            if (i + 1) % save_every_n == 0:
                save_data(all_outputs, all_targets, file, i) # all_inputs

    # Save final data
    save_data(all_outputs, all_targets, file, i) # all_inputs


def save_data(outputs, targets, file, batch_idx):  #inputs
    data = {
        'outputs': np.concatenate(outputs, 0),
        'targets': np.concatenate(targets, 0),
    }

    with open(file, 'wb') as f:
        pickle.dump(data, f)

    print(f'Saved data at batch {batch_idx}')



models = ['resnet50', 'vit_b_16','resnet50_init', 'vit_b_16_init', 'input']

for m in models:
    print(m)
    if 'resnet' in m:
        model = resnet50
        weights = ResNet50_Weights.DEFAULT
    elif 'vit' in m:
        model = vit_b_16
        weights = ViT_B_16_Weights.DEFAULT
    elif 'input' in m:
        model = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(224 * 224*3, 2048)  # same as resnet50
        )
        weights = ResNet50_Weights.DEFAULT
    else:
        raise ValueError(f"Model {m} not supported")

    # Step 1: Initialize model
    model = model(weights=None if 'init' in m else weights)
    model.eval()
    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()
    
    # Decapitate the model
    if 'resnet' in m:
        model.fc = torch.nn.Identity()
    elif 'vit' in m:
        model.heads = torch.nn.Identity()

    dataset = CustomImageDataset(samples=samples, transform=preprocess)
    
    # Example usage
    # Assuming model is your PyTorch model
    # Assuming dataset is your CustomImageDataset

    batch_size = 256
    save_every_n = 10  # Save every 100 batches

    # Define a DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # Process dataset and save results
    process_and_save(model, data_loader, os.path.join(SAVE_PATH, '%s.pkl' % m), save_every_n=save_every_n)