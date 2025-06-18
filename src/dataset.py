import h5py
import numpy as np
from torch.utils.data import Dataset
# import torchvision.transforms as transforms
import torch
import random
from processing import mu_law_compress
import torchio as tio
        
class BeamformData(Dataset):
    def __init__(self, path, normalization='max'):
        self.path = path
        self.normalization = normalization.strip().lower()
        self.db = None
        
        with h5py.File(self.path, 'r') as db: # open file once
            self.length = db["dwi"].shape[0]

    def __len__(self):
            return self.length

    def __getitem__(self, index):
        if self.db is None:
            self.db = h5py.File(self.path, 'r', swmr=True)

        try:
            dwi_data = self.db["dwi"][index]
            conv_data = self.db['tar'][index]

            dwi_data = torch.tensor(dwi_data, dtype=torch.cfloat)
            conv_data = torch.tensor(conv_data, dtype=torch.cfloat)

            if self.normalization == 'max':
                max_magnitude_dwi = torch.amax(torch.abs(dwi_data), dim=(1, 2, 3), keepdim=True)
                dwi_data /= max_magnitude_dwi
                max_magnitude_conv = torch.amax(torch.abs(conv_data), dim=(1, 2, 3), keepdim=True)
                conv_data /= max_magnitude_conv

            elif self.normalization == 'compand':
                max_magnitude_dwi = torch.amax(torch.abs(dwi_data), dim=(1, 2, 3), keepdim=True)
                dwi_data /= max_magnitude_dwi
                max_magnitude_conv = torch.amax(torch.abs(conv_data), dim=(1, 2, 3), keepdim=True)
                conv_data /= max_magnitude_conv
                dwi_data = mu_law_compress(dwi_data)
                conv_data = mu_law_compress(conv_data)

            elif self.normalization == 'mean_std':
                mean_dwi = torch.mean(dwi_data, dim=(1, 2, 3), keepdim=True)
                std_dwi = torch.std(dwi_data, dim=(1, 2, 3), keepdim=True) + 1e-6
                dwi_data = (dwi_data - mean_dwi) / std_dwi

                mean_conv = torch.mean(conv_data, dim=(1, 2, 3), keepdim=True)
                std_conv = torch.std(conv_data, dim=(1, 2, 3), keepdim=True) + 1e-6
                conv_data = (conv_data - mean_conv) / std_conv

            elif self.normalization == 'z_compand':
                mean_dwi = torch.mean(dwi_data, dim=(1, 2, 3), keepdim=True)
                std_dwi = torch.std(dwi_data, dim=(1, 2, 3), keepdim=True) + 1e-6
                dwi_data = (dwi_data - mean_dwi) / std_dwi

                mean_conv = torch.mean(conv_data, dim=(1, 2, 3), keepdim=True)
                std_conv = torch.std(conv_data, dim=(1, 2, 3), keepdim=True) + 1e-6
                conv_data = (conv_data - mean_conv) / std_conv

                dwi_data = mu_law_compress(dwi_data)
                conv_data = mu_law_compress(conv_data)

            elif self.normalization == 'std':
                std_dwi = torch.std(dwi_data, dim=(1, 2, 3), keepdim=True) + 1e-6
                dwi_data /= std_dwi
                std_conv = torch.std(conv_data, dim=(1, 2, 3), keepdim=True) + 1e-6
                conv_data /= std_conv

            else:
                raise ValueError(f"Unknown normalization type: {self.normalization}")

            # Check for NaNs or Infs
            if not torch.isfinite(dwi_data).all() or not torch.isfinite(conv_data).all():
                raise ValueError("NaN or Inf encountered")

            return dwi_data, conv_data

        except Exception as e:
            print(f"Skipping index {index} due to: {e}")
            # Try next index (wrap around if needed)
            new_index = (index + 1) % self.length
            return self.__getitem__(new_index)


class BeamformDataset(Dataset):
    def __init__(self, path, normalization='mean_std', transform=None, indices=None):
        self.path = path
        self.normalization = normalization.strip().lower()
        self.transform = transform
        self.db = None
        
        with h5py.File(self.path, 'r') as db: # open file once
            full_length = db["dwi"].shape[0]
        # If indices are provided, use them; else use full dataset
        self.indices = indices if indices is not None else list(range(full_length))

    def __len__(self):
            return len(self.indices)

    def __getitem__(self, index):
        if self.db is None:
            self.db = h5py.File(self.path, 'r', swmr=True)

        real_index = self.indices[index]

        dwi_data = self.db["dwi"][real_index]
        conv_data = self.db['tar'][real_index]

        dwi_data = torch.tensor(dwi_data, dtype=torch.cfloat)
        conv_data = torch.tensor(conv_data, dtype=torch.cfloat)

        try:
            if self.normalization == 'max':
                max_magnitude_dwi = torch.amax(torch.abs(dwi_data), dim=(1, 2, 3), keepdim=True)
                dwi_data /= max_magnitude_dwi
                max_magnitude_conv = torch.amax(torch.abs(conv_data), dim=(1, 2, 3), keepdim=True)
                conv_data /= max_magnitude_conv

            elif self.normalization == 'compand':
                max_magnitude_dwi = torch.amax(torch.abs(dwi_data), dim=(1, 2, 3), keepdim=True)
                dwi_data /= max_magnitude_dwi
                max_magnitude_conv = torch.amax(torch.abs(conv_data), dim=(1, 2, 3), keepdim=True)
                conv_data /= max_magnitude_conv
                dwi_data = mu_law_compress(dwi_data)
                conv_data = mu_law_compress(conv_data)

            elif self.normalization == 'mean_std':
                mean_dwi = torch.mean(dwi_data, dim=(1, 2, 3), keepdim=True)
                std_dwi = torch.std(dwi_data, dim=(1, 2, 3), keepdim=True) + 1e-6
                dwi_data = (dwi_data - mean_dwi) / std_dwi

                mean_conv = torch.mean(conv_data, dim=(1, 2, 3), keepdim=True)
                std_conv = torch.std(conv_data, dim=(1, 2, 3), keepdim=True) + 1e-6
                conv_data = (conv_data - mean_conv) / std_conv

            elif self.normalization == 'std':
                std_dwi = torch.std(dwi_data, dim=(1, 2, 3), keepdim=True) + 1e-6
                dwi_data /= std_dwi
                std_conv = torch.std(conv_data, dim=(1, 2, 3), keepdim=True) + 1e-6
                conv_data /= std_conv

            else:
                raise ValueError(f"Unknown normalization type: {self.normalization}")

            # Check for NaN or Inf values
            if not torch.isfinite(dwi_data).all() or not torch.isfinite(conv_data).all():
                raise ValueError("Invalid data (NaN or Inf)")

            if self.transform:
                dwi_data, conv_data = self.apply_joint_transform(dwi_data, conv_data)

            return dwi_data, conv_data

        except Exception as e:
            print(f"Skipping index {real_index} due to error: {e}")
            new_index = (index + 1) % len(self.indices)
            return self.__getitem__(new_index)

    
    def apply_joint_transform(self, dwi_volume, conv_volume):
        # dwi_volume, conv_volume: torch.cfloat, shape [C, D, H, W]

        def complex_to_real_channels(x):
            # [C, D, H, W] complex → [2*C, D, H, W] real
            real = x.real
            imag = x.imag
            stacked = torch.stack([real, imag], dim=1)  # [C, 2, D, H, W]
            return stacked.view(-1, *x.shape[1:])  # [2*C, D, H, W]

        def real_channels_to_complex(x, C):
            # [2*C, D, H, W] real → [C, D, H, W] complex
            restored = x.view(C, 2, *x.shape[1:])  # [C, 2, D, H, W]
            real = restored[:, 0]
            imag = restored[:, 1]
            return torch.complex(real, imag)  # [C, D, H, W]

        # Convert both to real tensors
        dwi_real = complex_to_real_channels(dwi_volume)
        conv_real = complex_to_real_channels(conv_volume)

        # Create a TorchIO Subject
        subject = tio.Subject(
            dwi=tio.Image(tensor=dwi_real, type=tio.INTENSITY),
            conv=tio.Image(tensor=conv_real, type=tio.INTENSITY),
        )

        transformed = self.transform(subject)

        # Convert back to complex
        C_dwi = dwi_volume.shape[0]
        C_conv = conv_volume.shape[0]
        dwi_transformed = real_channels_to_complex(transformed['dwi'].tensor, C_dwi)
        conv_transformed = real_channels_to_complex(transformed['conv'].tensor, C_conv)

        return dwi_transformed, conv_transformed
    
class BeamformDataset_patch_based(Dataset):
    def __init__(self, path, normalization='max',patch_size=64, transform=None, indices=None):
        self.path = path
        self.normalization = normalization.strip().lower()
        self.db = None
        self.patch_size = patch_size
        self.transform = transform
        
        with h5py.File(self.path, 'r') as db: # open file once
            full_length = db["dwi"].shape[0]
        self.indices = indices if indices is not None else list(range(full_length))

    def __len__(self):
            return len(self.indices)

    def __getitem__(self, index):
        # open file once per worker
        if self.db is None:
             self.db = h5py.File(self.path, 'r', swmr=True)
        
        real_index = self.indices[index]

        # lazy indexing
        dwi_data = torch.tensor(self.db["dwi"][real_index], dtype=torch.cfloat)
        conv_data = torch.tensor(self.db["tar"][real_index], dtype=torch.cfloat)

        #Patch based approach
        D, H, W = dwi_data.shape[-3:]
        z = random.randint(0, D - self.patch_size)
        y = random.randint(0, H - self.patch_size)
        x = random.randint(0, W - self.patch_size)
        dwi_data = dwi_data[:, z:z+self.patch_size, y:y+self.patch_size, x:x+self.patch_size]
        conv_data = conv_data[:, z:z+self.patch_size, y:y+self.patch_size, x:x+self.patch_size]


        if self.normalization == 'max':
            max_magnitude_dwi = torch.amax(torch.abs(dwi_data), dim=(1, 2, 3), keepdim=True)
            dwi_data /= max_magnitude_dwi
            max_magnitude_conv = torch.amax(torch.abs(conv_data), dim=(1, 2, 3), keepdim=True)
            conv_data /= max_magnitude_conv

        elif self.normalization == 'compand':
            max_magnitude_dwi = torch.amax(torch.abs(dwi_data), dim=(1, 2, 3), keepdim=True)
            dwi_data /= max_magnitude_dwi
            max_magnitude_conv = torch.amax(torch.abs(conv_data), dim=(1, 2, 3), keepdim=True)
            conv_data /= max_magnitude_conv
            dwi_data = mu_law_compress(dwi_data)
            conv_data = mu_law_compress(conv_data)
        
        elif self.normalization == 'mean_std':
            mean_dwi = torch.mean(dwi_data, dim=(1, 2, 3), keepdim=True)
            std_dwi = torch.std(dwi_data, dim=(1, 2, 3), keepdim=True) + 1e-6  # Avoid division by zero
            dwi_data = (dwi_data - mean_dwi) / std_dwi
            # Apply the same to conv_data (target)
            mean_conv = torch.mean(conv_data, dim=(1, 2, 3), keepdim=True)
            std_conv = torch.std(conv_data, dim=(1, 2, 3), keepdim=True) + 1e-6
            conv_data = (conv_data - mean_conv) / std_conv
        
        elif self.normalization == 'std':
                std_dwi = torch.std(dwi_data, dim=(1, 2, 3), keepdim=True) + 1e-6
                dwi_data /= std_dwi
                std_conv = torch.std(conv_data, dim=(1, 2, 3), keepdim=True) + 1e-6
                conv_data /= std_conv

        else:
            raise ValueError(f"Unknown normalization type: {self.normalization}")
        
        if self.transform:
            dwi_data = tio.Image(tensor=dwi_data, type=tio.INTENSITY)
            dwi_data = self.transform(dwi_data)
            dwi_data = dwi_data.tensor

            conv_data = tio.Image(tensor=conv_data, type=tio.INTENSITY)
            conv_data = self.transform(conv_data)
            conv_data = conv_data.tensor
            # dwi_data, conv_data = self.transform(dwi_data, conv_data)
        
        return dwi_data, conv_data
    
