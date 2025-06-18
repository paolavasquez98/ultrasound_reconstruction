import skimage
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_msssim import SSIM
from sklearn.feature_selection import mutual_info_regression
from processing import compress_iq
import torch.nn as nn

class ComplexMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(ComplexMSELoss, self).__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # Compute squared difference on real and imaginary parts
        loss = (y_true.real - y_pred.real) ** 2 + (y_true.imag - y_pred.imag) ** 2

        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:  # 'none'
            return loss
        
class ComplexL1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(ComplexL1Loss, self).__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # Compute absolute difference on real and imaginary parts
        loss = torch.abs(y_true.real - y_pred.real) + torch.abs(y_true.imag - y_pred.imag)

        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:  # 'none'
            return loss
        
class ComplexTVLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(ComplexTVLoss, self).__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction

    def forward(self, x):
        # TV on real part
        real = x.real
        imag = x.imag

        loss_real = (
            torch.abs(real[:, :, :-1, :, :] - real[:, :, 1:, :, :]).sum() +
            torch.abs(real[:, :, :, :-1, :] - real[:, :, :, 1:, :]).sum() +
            torch.abs(real[:, :, :, :, :-1] - real[:, :, :, :, 1:]).sum()
        )

        # TV on imaginary part
        loss_imag = (
            torch.abs(imag[:, :, :-1, :, :] - imag[:, :, 1:, :, :]).sum() +
            torch.abs(imag[:, :, :, :-1, :] - imag[:, :, :, 1:, :]).sum() +
            torch.abs(imag[:, :, :, :, :-1] - imag[:, :, :, :, 1:]).sum()
        )

        loss = loss_real + loss_imag

        if self.reduction == 'mean':
            return loss / x.numel()
        elif self.reduction == 'sum':
            return loss
        else:  # 'none'
            return loss 

class ComplexL1TVLoss(nn.Module):
    def __init__(self, tv_weight=0.01, reduction='mean'):
        super().__init__()
        self.l1 = ComplexL1Loss(reduction=reduction)
        self.tv = ComplexTVLoss(reduction=reduction)
        self.tv_weight = tv_weight

    def forward(self, y_pred, y_true):
        l1_loss = self.l1(y_pred, y_true)
        tv_loss = self.tv(y_pred)
        return l1_loss + self.tv_weight * tv_loss

class ComplexMSETVLoss(nn.Module):
    def __init__(self, tv_weight=0.01, reduction='mean'):
        super().__init__()
        self.mse = ComplexMSELoss(reduction=reduction)
        self.tv = ComplexTVLoss(reduction=reduction)
        self.tv_weight = tv_weight

    def forward(self, y_pred, y_true):
        mse_loss = self.mse(y_pred, y_true)
        tv_loss = self.tv(y_pred)
        return mse_loss + self.tv_weight * tv_loss
    
class ComplexMSE_SSIMLoss(nn.Module):
    def __init__(self, alpha=1, beta=0.5,  reduction='mean'):
        super(ComplexMSE_SSIMLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = ComplexMSELoss(reduction=reduction)
        self.ssim_module = SSIM(data_range=1.0, size_average=True, channel=1, spatial_dims=3)

    def forward(self, y_pred, y_true):
        # MSE on complex-valued data
        mse = self.mse_loss(y_pred, y_true)

        # SSIM on envelope
        y_pred_mag = y_pred.abs()
        y_true_mag = y_true.abs()
        ssim_loss = 1.0 - (self.ssim_module(y_pred_mag, y_true_mag))

        # Total combined loss
        total_loss = self.alpha * mse + self.beta * ssim_loss
        return total_loss
    
# metrics fro validation and testing -------------------
def get_psnr(y_pred, y_true, g_compr=False, return_per_sample=False):
    """Gamma compression of the pred and true, by default calculates on the log compressed image"""
    if g_compr:
        y_pred = compress_iq(y_pred, mode='log')
        y_true = compress_iq(y_true, mode='log')
    else: 
        y_pred = y_pred.abs() 
        y_true = y_true.abs()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu()

    y_pred_np = y_pred.cpu().numpy()
    y_true_np = y_true.cpu().numpy()

    psnr_values = []
    for i in range(y_pred.shape[0]):
        pred_i = y_pred_np[i,0] # get the volume [192,192,192]
        true_i = y_true_np[i,0]
        psnr = skimage.metrics.peak_signal_noise_ratio(true_i, pred_i, data_range=np.max(true_i))
        psnr_values.append(psnr)
    if return_per_sample:
        return psnr_values
    else:
        return np.mean(psnr_values)


def get_ssim(y_pred, y_true, g_compr=False):
    """Calculate the ssim for 3D images per batch, by first doing lo compression or get hte magnitud
    input size must be tensors [B, 1, D, H, W]
    works for validation only (because data was detahced)
    https://github.com/VainF/pytorch-msssim"""
    if g_compr:
        # Perform log compression 
        y_pred = compress_iq(y_pred, mode='log')
        y_true = compress_iq(y_true, mode='log')
    else:
        y_pred = y_pred.abs()
        y_true = y_true.abs()

    B, C, D, H, W = y_pred.shape
    ssim_scores = []
    ssim_module = SSIM(data_range=1.0, size_average=True, channel=1, spatial_dims=3)

    for b in range(B):
        ssim_val = ssim_module(y_pred[b:b+1], y_true[b:b+1])  # shape: [1, 1, D, H, W]
        ssim_scores.append(ssim_val.item())

    return sum(ssim_scores) / B

# Mutual Information
def mutual_info(y_pred, y_true):
    y_pred = y_pred.abs().cpu().numpy().flatten()
    y_true = y_true.abs().cpu().numpy().flatten()
    return mutual_info_regression(y_true.reshape(-1, 1), y_pred)[0]


def plot_slice(volume, slice_idx=None):
    """Plots a slice of volume """
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy() 
    assert volume.ndim == 3, "Input volume must be 3D (Depth, Height, Width)"
    depth, height, width = volume.shape  # [192, 192, 192]
    if slice_idx is None:
        slice_idx = width // 2  # Select the middle slice by default
    side_slice = volume[:, slice_idx, :]  # Extract the sagittal slice
    # Plot the side view
    plt.figure(figsize=(6, 6))
    plt.imshow(side_slice, cmap='gray', aspect='auto')
    plt.axis('off')
    plt.title(f"Volume Slice {slice_idx}")
    plt.show()

