import argparse
import json
import sys
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

from utils import load_data, plot_loss_curve, EarlyStopCriterion

from tqdm import tqdm

BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULT_FOLDER = "cnn_%Y_%m_%d_%H_%M_%S"
INPUT_HEIGHT = 144
INPUT_WIDTH = 224

# CNN model running on the car
class CNN(nn.Module):
    """
    CNN using kernel sizes from the diagram:
    5x5, 5x5, 5x5, 3x3, 3x3
    No image resizing.
    Input: (N, 3, 144, 224)
    Output: (N, 1)
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        # For input 3x144x224 → conv stack → 64x11x21
        self.flatten_dim = 64 * 11 * 21

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 100),
            nn.ReLU(inplace=True),

            nn.Linear(100, 50),
            nn.ReLU(inplace=True),

            nn.Linear(50, 10),
            nn.ReLU(inplace=True),

            nn.Linear(10, 1),
        )

    def forward(self, x):
        # Normalize to [-1,1] as in NVIDIA paper
        x = x.permute(0,3,1,2).float()
        # Resize collected images to the architecture's expected input size.
        if x.shape[-2:] != (INPUT_HEIGHT, INPUT_WIDTH):
            x = F.interpolate(
                x,
                size=(INPUT_HEIGHT, INPUT_WIDTH),
                mode='bilinear',
                align_corners=False
            )
        x = x / 255.0
        x = x * 2.0 - 1.0

        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)
    
def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer
    ):
    model.train()

    total_loss = 0
    for image, steer in dataloader:
        pred_steer = model(image)
        loss = criterion(pred_steer, steer).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    return avg_loss

def val_epoch(
        model: nn.Module,
        dataloader : DataLoader,
        criterion: nn.Module
    ):
    model.eval()
    total_loss = 0
    for image, steer in dataloader:
        with torch.no_grad():
            pred_steer = model(image)
            loss = criterion(pred_steer, steer).mean()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train(
        model: nn.Module,
        images: torch.Tensor, 
        steers: torch.Tensor, 
        device = DEVICE,
        batch_size = BATCH_SIZE,
        nepochs = NUM_EPOCHS,
        lr = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY
    ):
    device = torch.device(device)
    
    images = images.to(device, dtype=torch.float32)
    steers = steers.to(device, dtype=torch.float32)

    dataset = TensorDataset(images, steers)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    model = model.to(device)  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss(reduction='none')

    pbar = tqdm(range(1, nepochs+1), desc="Training CNN Controller")

    for epoch in pbar:
        
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer)
        val_loss = val_epoch(model, val_dataloader, criterion)

        pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

        yield train_loss, val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config", type=str)

    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Failed to load config file {args.config}: {e}")
        sys.exit(1)

    try:
        data = load_data(config['load_data'])
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)

    try:
        training_data_config = config['training_data']

        steer_list = []
        image_list = []
        for k, v in training_data_config.items():

            steer_list.append(data[k]['steer'][slice(*v)])
            image_list.append(data[k]['image'][slice(*v)])

        images = torch.from_numpy(np.concatenate(image_list))
        steers = torch.from_numpy(np.concatenate(steer_list)).unsqueeze(1)

    except Exception as e:
        print(f"Failed to construct training data: {e}")
        sys.exit(1)

    print(f"Training data shape: images {list(images.shape)} steer {list(steers.shape)}")

    result_folder = Path(datetime.now().strftime(config.get('result_folder', RESULT_FOLDER)))
    result_folder.mkdir(parents=True, exist_ok=True)

    train_loss_list = []
    val_loss_list = []

    early_stop = EarlyStopCriterion()

    model = CNN()

    train_config = config.get('train', {})

    for epoch, (train_loss, val_loss) in enumerate(train(model,
                                                        images, 
                                                        steers,
                                                        **train_config),1):

        if early_stop(train_loss, val_loss):
            torch.save(model.state_dict(), result_folder / 'best.pt')
            tqdm.write(f"Saved Model in epoch {epoch}")

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

    fig, ax = plot_loss_curve(train_loss_list, val_loss_list)
    fig.savefig(result_folder / "loss_curve.png", dpi=200)


