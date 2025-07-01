import os
import sys
import torch
import numpy as np
import random
import wandb
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, "src"))

from dataset import BeamformDataset, BeamformDataset_patch_based, BeamformData
from utils import ComplexL1TVLoss, ComplexMSETVLoss, ComplexMSE_SSIMLoss, ComplexMSELoss
from model import CIDNet3D, CxUnet_RB, ComplexRNN_GRU_all_h, ComplexRNN_GRU
from training import train_and_validate
from testing import test_model

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


wandb.init(
    entity="paolavasquez98",
    project="us_reconstruction",
    dir='/home/vasquez/python_part/beamforming/wandb_logs',
    name="CIDNet3D_big_data",
    job_type="train-test",
    # group="final data",
    save_code=True,
)

# --------------------- Dataset and Dataloader Setup ---------------------
normalization = 'std'  # 'compand', 'max', 'mean_std'
data_path = '/home/vasquez/python_part/dataset_creation/results/dataset_hr_train.h5'
dataset = BeamformData(data_path, normalization=normalization)

# Split data
idx = list(range(len(dataset)))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

# Subsets
train_data = Subset(dataset, train_idx)
val_data = Subset(dataset, test_idx)

# Parameters
batch_size = 2
epochs = 100
patience = 7
learning_rate = 1e-4
min_lr = 1e-6

# Dataloaders
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True)

# --------------------- Model Setup ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device {device}")

model = CIDNet3D().to(device)
model_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
criterion = ComplexMSETVLoss()
# criterion = ComplexMSE_SSIMLoss(alpha=1.0, beta=0.5)

wandb.config.update({
    "epochs": epochs,
    "optimizer": optimizer.__class__.__name__,
    "learning_rate": learning_rate,
    "batch_size": train_dataloader.batch_size,
    "accumulation_steps": 2,
    "loss": criterion.__class__.__name__,
    "architecture": model.__class__.__name__, 
    "data": os.path.basename(data_path),
    "normalization": normalization,
    "model_parameters": model_params,
    "train_data_size": len(train_data),
}, allow_val_change=True)
wandb.watch(model, log="all", log_freq=100)

print("All configurations logged to wandb")

# Train and Validate
training_results, best_model_path = train_and_validate(model, train_dataloader, val_dataloader, optimizer, criterion, scheduler, epochs, device, patience)

print(f"Experiment configuration saved to wandb: {wandb.run.dir}")

# ---------------------- Testing ---------------------
test_path = '/home/vasquez/python_part/dataset_creation/results/dataset_hr_test.h5'
test_dataset = BeamformData(test_path, normalization=normalization)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False) 

def load_best_model_if_exists(model, path, device):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded best model from {path}")
    else:
        print(f"Warning: Best model file not found at {path}")


load_best_model_if_exists(model, best_model_path, device)

loss, psnr, ssim, images = test_model(model, test_dataloader, criterion, device, normalization)

for i, img_dict in enumerate(images):
    wandb.log({
        f"Sample_{i}": [
            wandb.Image(img_dict["image"], caption="Input/Prediction/GT (raw)"),
            wandb.Image(img_dict["log_image"], caption="Log compressed"),
        ]
    })

wandb.finish()