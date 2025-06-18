import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import get_psnr, get_ssim
from processing import compress_iq, mu_law_expand

def plot_input_pred_tar(input_img, prediction, target):
    depth_idx = input_img.shape[0] // 2  # Middle depth slice

    # Extract the same depth slice across all three volumes
    input_slice = input_img[:, :, depth_idx]
    pred_slice = prediction[:, :, depth_idx]
    target_slice = target[:, :, depth_idx]

    # Create the subplot grid
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot Input
    axes[0].imshow(input_slice, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title("Input")

    # Plot Prediction
    axes[1].imshow(pred_slice, cmap='gray')
    axes[1].axis('off')
    axes[1].set_title("Prediction")

    # Plot Target
    axes[2].imshow(target_slice, cmap='gray')
    axes[2].axis('off')
    axes[2].set_title("Target")

    plt.tight_layout()
    return fig

def test_model(model, dataloader, criterion, device, norm_method):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    psnr_values = []
    ssim_values = []
    image_logs = []

    with torch.no_grad():
        tqdm_bar = tqdm(dataloader, desc="Testing", leave=False)
        
        for batch_idx, (dwi_data, conv_data) in enumerate(tqdm_bar):
            dwi_data, conv_data = dwi_data.to(device), conv_data.to(device)
            # dwi_data Shape: [batch_size, 9, 192, 192, 192]
            # conv_data Shape: [batch_size, 1, 192, 192, 192]

            # Forward pass
            outputs = model(dwi_data) # output Shape: [batch_size, 1, 192, 192, 192]

            # Save some data for visualization
            if batch_idx % 4 == 0 and len(image_logs) < 10:

                if norm_method == 'compand':
                    pred_img = outputs#.cpu().numpy()
                    gt_img = conv_data#.cpu().numpy()
                    input_img = dwi_data#.cpu().numpy()

                    expanded_input = mu_law_expand(input_img).cpu().numpy()
                    inp = np.mean(expanded_input[0], axis=0).squeeze()
                    expanded_pred = mu_law_expand(pred_img).cpu().numpy()
                    pred = expanded_pred[0, 0].squeeze()
                    expanded_tar = mu_law_expand(gt_img).cpu().numpy()
                    tar = expanded_tar[0, 0].squeeze()
                    img = plot_input_pred_tar(np.abs(inp), np.abs(pred), np.abs(tar))
                    log_img = plot_input_pred_tar(
                    compress_iq(inp, mode='log', dynamic_range_dB=60),
                    compress_iq(pred, mode='log', dynamic_range_dB=60),
                    compress_iq(tar, mode='log', dynamic_range_dB=60))
                elif norm_method == 'max':
                    pred_img = outputs.cpu().numpy()
                    gt_img = conv_data.cpu().numpy()
                    input_img = dwi_data.cpu().numpy()

                    inp = np.mean(input_img[0], axis=0).squeeze()
                    pred = pred_img[0, 0].squeeze()
                    tar = gt_img[0, 0].squeeze()
                    img = plot_input_pred_tar(np.abs(inp), np.abs(pred), np.abs(tar))
                    log_img = plot_input_pred_tar(
                    compress_iq(inp, mode='log', dynamic_range_dB=60),
                    compress_iq(pred, mode='log', dynamic_range_dB=60),
                    compress_iq(tar, mode='log', dynamic_range_dB=60))
                elif norm_method == 'mean_std':
                    pred_img = outputs.cpu().numpy()
                    gt_img = conv_data.cpu().numpy()
                    input_img = dwi_data.cpu().numpy()
                    inp = np.mean(input_img[0], axis=0).squeeze()
                    pred = pred_img[0, 0].squeeze()
                    tar = gt_img[0, 0].squeeze()
                    img = plot_input_pred_tar(np.abs(inp), np.abs(pred), np.abs(tar))
                    log_img = plot_input_pred_tar(
                    compress_iq(inp, mode='log', dynamic_range_dB=60),
                    compress_iq(pred, mode='log', dynamic_range_dB=60),
                    compress_iq(tar, mode='log', dynamic_range_dB=60))
                elif norm_method == 'z_compand':
                    pred_img = outputs#.cpu().numpy()
                    gt_img = conv_data#.cpu().numpy()
                    input_img = dwi_data#.cpu().numpy()
                    expanded_input = mu_law_expand(input_img).cpu().numpy()
                    inp = np.mean(expanded_input[0], axis=0).squeeze()
                    expanded_pred = mu_law_expand(pred_img).cpu().numpy()
                    pred = expanded_pred[0, 0].squeeze()
                    expanded_tar = mu_law_expand(gt_img).cpu().numpy()
                    tar = expanded_tar[0, 0].squeeze()
                    img = plot_input_pred_tar(np.abs(inp), np.abs(pred), np.abs(tar))
                    log_img = plot_input_pred_tar(
                    compress_iq(inp, mode='log', dynamic_range_dB=60),
                    compress_iq(pred, mode='log', dynamic_range_dB=60),
                    compress_iq(tar, mode='log', dynamic_range_dB=60))
                elif norm_method == 'std':
                    pred_img = outputs.cpu().numpy()
                    gt_img = conv_data.cpu().numpy()
                    input_img = dwi_data.cpu().numpy()
                    inp = np.mean(input_img[0], axis=0).squeeze()
                    pred = pred_img[0, 0].squeeze()
                    tar = gt_img[0, 0].squeeze()
                    img = plot_input_pred_tar(np.abs(inp), np.abs(pred), np.abs(tar))
                    log_img = plot_input_pred_tar(
                    compress_iq(inp, mode='log', dynamic_range_dB=60),
                    compress_iq(pred, mode='log', dynamic_range_dB=60),
                    compress_iq(tar, mode='log', dynamic_range_dB=60))
                else:
                    raise ValueError(f"Unknown normalization method: {norm_method}")
                

                # Save slices for visualization
                image_logs.append({
                "image": img,
                "log_image": log_img,
                # "gamma_image": gamma_img
                })

            # Compute loss
            loss = criterion(outputs, conv_data)
            running_loss += loss.item()

            # --- ADD THESE PRINT STATEMENTS ---
            print(f"Number of batches processed: {len(dataloader)}")
            print(f"Length of psnr_values: {len(psnr_values)}")
            print(f"Length of ssim_values: {len(ssim_values)}")
            print(f"Running loss at end of loop: {running_loss}")
            # -----------------------------------

            # Compute PSNR and SSIM
            psnr_value = get_psnr(outputs, conv_data)
            ssim_value = get_ssim(outputs, conv_data)
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)

            tqdm_bar.set_postfix(
                loss=running_loss / (batch_idx + 1),
                psnr=np.mean(psnr_values),
                ssim=np.mean(ssim_values)
        )
            wandb.log({
                "test_loss_mse": loss,
                "test_psnr": psnr_value,
                "test_ssim": ssim_value
            })

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_loss = running_loss / len(dataloader)
    print(f"\nTest Loss: {avg_loss:.4f}")
    print(f"\nTest PSNR: {avg_psnr:.2f} dB")
    print(f"\nTest SSIM: {avg_ssim:.4f}")
    # wandb.log({
    #             "test_loss_mse": avg_loss,
    #             "test_psnr": avg_psnr,
    #             "test_ssim": avg_ssim
    #         })
    return avg_loss, avg_psnr, avg_ssim, image_logs