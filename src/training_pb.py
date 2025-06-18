import torch
import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import get_psnr, get_ssim
from processing import compress_iq, mu_law_expand

def plot_input_pred_tar(input_img, prediction, target):
    width_idx = input_img.shape[2] // 2  # sagittal slice (W axis)

    input_slice = input_img[:, :, width_idx]
    pred_slice = prediction[:, :, width_idx]
    target_slice = target[:, :, width_idx]

    # Create subplot grid
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Set voxel spacing: z=0.5, y=1.0 (Depth × Height)
    aspect_ratio = 0.5 / 1.0  # = 0.5

    for ax, img, title in zip(axes, [input_slice, pred_slice, target_slice], ["Input", "Prediction", "Target"]):
        ax.imshow(img, cmap='gray', aspect=aspect_ratio)
        ax.axis('off')
        ax.set_title(title)

    plt.tight_layout()
    return fig

def sliding_window_inference(
    model: torch.nn.Module,
    input_volume: torch.Tensor, # This should be a single full volume (C, D, H, W)
    patch_size: tuple[int, int, int],
    overlap_ratio: float = 0.5,
    batch_size_per_window: int = 1, # Max patches to process in parallel on GPU
    mode: str = 'gaussian',  # 'gaussian', 'constant' (for averaging)
    sigma_scale: float = 0.125, # Scale for Gaussian weighting, affects how quickly weights drop off
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> torch.Tensor:

    model.eval() # Set model to evaluation mode
    model.to(device)

    # Add batch dimension to the single input volume: (C, D, H, W) -> (1, C, D, H, W)
    if input_volume.ndim == 4:
        input_volume = input_volume.unsqueeze(0)
    elif input_volume.ndim != 5:
        raise ValueError(f"Input volume must be 4D (C,D,H,W) or 5D (1,C,D,H,W), but got {input_volume.ndim}D.")

    _, C_in, D_full, H_full, W_full = input_volume.shape

    # Calculate strides based on overlap
    stride_d = int(patch_size[0] * (1 - overlap_ratio))
    stride_h = int(patch_size[1] * (1 - overlap_ratio))
    stride_w = int(patch_size[2] * (1 - overlap_ratio))

    # Ensure strides are at least 1
    stride_d = max(1, stride_d)
    stride_h = max(1, stride_h)
    stride_w = max(1, stride_w)

    # Initialize output prediction and count maps
    accumulated_prediction = None
    contribution_map = None

    # Generate patch coordinates
    patches_coords = []
    for d_start in range(0, D_full, stride_d):
        for h_start in range(0, H_full, stride_h):
            for w_start in range(0, W_full, stride_w):
                d_end = min(d_start + patch_size[0], D_full)
                h_end = min(h_start + patch_size[1], H_full)
                w_end = min(w_start + patch_size[2], W_full)

                # Adjust start coordinates if the patch goes out of bounds
                d_start = d_end - patch_size[0] if d_end == D_full and d_start + patch_size[0] > D_full else d_start
                h_start = h_end - patch_size[1] if h_end == H_full and h_start + patch_size[1] > H_full else h_start
                w_start = w_end - patch_size[2] if w_end == W_full and w_start + patch_size[2] > W_full else w_start
                
                # Ensure start coordinates are non-negative
                d_start = max(0, d_start)
                h_start = max(0, h_start)
                w_start = max(0, w_start)

                patches_coords.append((d_start, d_end, h_start, h_end, w_start, w_end))

    # Create Gaussian weights if mode is 'gaussian'
    gaussian_kernel = None
    if mode == 'gaussian':
        sigma = [s * sigma_scale for s in patch_size] # Sigma relative to patch size
        center_d = patch_size[0] // 2
        center_h = patch_size[1] // 2
        center_w = patch_size[2] // 2
        
        # Create a 3D Gaussian kernel
        coords_d = torch.arange(patch_size[0], dtype=torch.float32, device=device) - center_d
        coords_h = torch.arange(patch_size[1], dtype=torch.float32, device=device) - center_h
        coords_w = torch.arange(patch_size[2], dtype=torch.float32, device=device) - center_w
        
        d_grid, h_grid, w_grid = torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij')
        
        gaussian_kernel = torch.exp(
            -( (d_grid**2 / (2 * sigma[0]**2)) + 
               (h_grid**2 / (2 * sigma[1]**2)) + 
               (w_grid**2 / (2 * sigma[2]**2)) )
        )
        gaussian_kernel = gaussian_kernel / gaussian_kernel.max() # Normalize to max 1

    # Process patches in batches
    num_patches = len(patches_coords)
    
    with torch.no_grad(): # Essential for inference to save memory and speed up
        for i in range(0, num_patches, batch_size_per_window):
            batch_coords = patches_coords[i : i + batch_size_per_window]
            
            # Extract actual patches for the batch
            current_batch_patches = []
            for d_s, d_e, h_s, h_e, w_s, w_e in batch_coords:
                current_batch_patches.append(input_volume[:, :, d_s:d_e, h_s:h_e, w_s:w_e])
            
            # Concatenate patches along the batch dimension
            current_batch_patches_tensor = torch.cat(current_batch_patches, dim=0).to(device)

            # Perform model inference
            batch_predictions = model(current_batch_patches_tensor) 
            # batch_predictions shape: (batch_size_per_window, Num_classes, patch_d, patch_h, patch_w)

            # Initialize accumulated tensors on first prediction
            if accumulated_prediction is None:
                num_classes_out = batch_predictions.shape[1]
                # Accumulated prediction and contribution map will match the full volume shape
                output_volume_shape = (num_classes_out, D_full, H_full, W_full)
                accumulated_prediction = torch.zeros(output_volume_shape, dtype=batch_predictions.dtype, device=device)
                contribution_map = torch.zeros(output_volume_shape, dtype=torch.float32, device=device)
            
            for j, (d_s, d_e, h_s, h_e, w_s, w_e) in enumerate(batch_coords):
                patch_prediction = batch_predictions[j] # (Num_classes, patch_d, patch_h, patch_w)
                
                if mode == 'gaussian':
                    # Apply Gaussian weighting to the patch prediction
                    weighted_patch_prediction = patch_prediction * gaussian_kernel
                    accumulated_prediction[:, d_s:d_e, h_s:h_e, w_s:w_e] += weighted_patch_prediction
                    contribution_map[:, d_s:d_e, h_s:h_e, w_s:w_e] += gaussian_kernel
                else: # Constant weighting
                    accumulated_prediction[:, d_s:d_e, h_s:h_e, w_s:w_e] += patch_prediction
                    contribution_map[:, d_s:d_e, h_s:h_e, w_s:w_e] += 1.0 # Increment count for averaging

    final_prediction = accumulated_prediction / (contribution_map + 1e-6)
    final_prediction = final_prediction.unsqueeze(0)

    return final_prediction

# TRAIN ONE EPOCH
def train(model, train_loader, optimizer, criterion, device, save_predictions=False, num_samples=5):
    model.train()  # Set model to training mode
    epoch_loss = 0.0
    saved_data = 0
    image_logs = []

    for batch_idx, (dwi_data, conv_data) in enumerate(tqdm(train_loader, desc="Training")):
        dwi_data, conv_data = dwi_data.to(device), conv_data.to(device)

        train_out = model(dwi_data)
        train_loss = criterion(train_out, conv_data)
        
        train_loss.backward()
        optimizer.step()

        epoch_loss += train_loss.item()
        optimizer.zero_grad()  

        # save some predictios
        if save_predictions and saved_data < num_samples:
            pred_img = train_out[0].detach().cpu().numpy()
            pred_img = compress_iq(pred_img.squeeze(0), mode='log', dynamic_range_dB=60)
            gt_img = conv_data[0].detach().cpu().numpy()
            gt_img = compress_iq(gt_img.squeeze(0), mode='log', dynamic_range_dB=60)
            input_img = dwi_data[0].detach().cpu().numpy()
            input_img = np.mean(input_img, axis=0, keepdims=True)
            input_img = compress_iq(input_img.squeeze(0), mode='log', dynamic_range_dB=60)

            slice = input_img.shape[0] // 2
            input_slice = input_img[:, :, slice]
            pred_slice = pred_img[:, :, slice]
            gt_slice = gt_img[:, :, slice]

            image_logs.append({
                "input": input_slice,
                "prediction": pred_slice,
                "ground_truth": gt_slice
            })
            saved_data += 1


    return epoch_loss / len(train_loader), image_logs  # Return average loss

# VALIDATION
def validate(model, val_loader, criterion, device, epoch, patch_size=(96,96,96), batch_size_per_window=4):
    model.eval()  # Set model to evaluation mode
    epoch_loss = 0.0
    psnr_values = []
    ssim_values = []
    with torch.no_grad():  # No gradient computation for validation
        for batch_idx, (dwi_data, conv_data) in enumerate(tqdm(val_loader, desc="Validating")):
            dwi_data, conv_data = dwi_data.to(device), conv_data.to(device)

            # Forward pass
            val_out = sliding_window_inference(
                model=model,
                input_volume=dwi_data, # This is a single (C, D, H, W) volume
                patch_size=patch_size, 
                overlap_ratio=0.5,       
                batch_size_per_window=batch_size_per_window, 
                mode='gaussian',       
                device=device
            )

            # Compute loss
            val_loss = criterion(val_out, conv_data)
            epoch_loss += val_loss.item()

            # Compute PSNR and SSIM per batch
            psnr_value = get_psnr(val_out, conv_data)
            ms_ssim_value = get_ssim(val_out, conv_data)

            psnr_values.append(psnr_value)
            ssim_values.append(ms_ssim_value)

    # average for the epoch
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    return epoch_loss / len(val_loader), avg_psnr, avg_ssim 


# FULL TRAINING FUNCTION
def train_and_validate(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs, device, patience, patch_size, batch_size_per_window):
    best_val_loss = float("inf")  # Initialize with a high value
    best_model_path = os.path.join(wandb.run.dir, 'best_model.pth')
    early_stopping_counter = 0
    best_image_logs = []
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        train_loss, image_logs = train(model, train_loader, optimizer, criterion, device, save_predictions=True, num_samples=5)
        val_loss, avg_psnr, avg_ssim = validate(model, val_loader, criterion, device, epoch, patch_size, batch_size_per_window)
        scheduler.step()

        # Log everything to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_psnr": avg_psnr,
            "val_ssim": avg_ssim,
        })

        print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, PSNR = {avg_psnr:.2f} dB, SSIM = {avg_ssim:.3f}")
        if val_loss < best_val_loss:
            # Save images 
            best_image_logs = image_logs[:5]  # Save for later
            image_logs = []  # Clear after saving, to avoid old data mixing
            for i, img_dict in enumerate(best_image_logs):
                wandb.log({
                    f"BestSample_E{epoch}_#{i}": [
                        wandb.Image(img_dict["input"], caption="Input"),
                        wandb.Image(img_dict["prediction"], caption="Prediction"),
                        wandb.Image(img_dict["ground_truth"], caption="Ground Truth")
                    ]
                })

            print(f"New best model found at epoch {epoch} with Val Loss: {val_loss:.6f}")
            best_val_loss = val_loss
            best_psnr = avg_psnr
            best_ssim = avg_ssim
            # Save the best model
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")
            wandb.save(best_model_path)  # optional: upload model to wandb
            wandb.config.update({"best_model_path": best_model_path})
            artifact = wandb.Artifact(f"{wandb.run.name}-best-model", type="model")
            artifact.add_file(best_model_path)
            wandb.log_artifact(artifact)

            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_psnr"] = best_psnr
            wandb.run.summary["best_ssim"] = best_ssim
            print(f"\nBest Val Loss: {best_val_loss:.6f} | PSNR: {best_psnr:.2f} dB | SSIM: {best_ssim:.3f}")
            early_stopping_counter = 0  # Reset counter
        else:
            early_stopping_counter += 1
            print(f"No improvement in validation loss. Early stopping counter: {early_stopping_counter}/{patience}")
            if early_stopping_counter >= patience:
                wandb.run.summary["stopped_epoch"] = epoch
                print(f"Stopping early after {epoch} epochs due to no improvement.")
                break

    print("\nTraining complete!")
    return best_image_logs, best_model_path

def test_model(model, dataloader, criterion, device, norm_method, patch_size=96, batch_size_per_window=4):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    psnr_values = []
    ssim_values = []
    image_logs = []

    with torch.no_grad():
        tqdm_bar = tqdm(dataloader, desc="Testing", leave=False)
        
        for batch_idx, (dwi_data, conv_data) in enumerate(tqdm_bar):
            dwi_data, conv_data = dwi_data.to(device), conv_data.to(device)

            # Forward pass
            outputs = sliding_window_inference(
                model=model,
                input_volume=dwi_data, # This is a single (C, D, H, W) volume
                patch_size=(patch_size, patch_size, patch_size), # Your model's expected patch size
                overlap_ratio=0.5,       
                batch_size_per_window=batch_size_per_window, # Adjust based on GPU memory. How many patches can be processed concurrently during inference.
                mode='gaussian',         # Recommended for smoother results
                device=device
            )

            # Save some data for visualization
            if batch_idx % 4 == 0 and len(image_logs) < 10:
                if norm_method == 'max':
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
                })

            # Compute loss
            loss = criterion(outputs, conv_data)
            running_loss += loss.item()

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
    return avg_loss, avg_psnr, avg_ssim, image_logs
