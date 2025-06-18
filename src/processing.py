import torch
import numpy as np
import torch


def compress_iq(iq_data, mode='log', dynamic_range_dB=50, gamma=0.5):
    """works for 3D tensors and numpy arrays and batches as well"""
    is_tensor = isinstance(iq_data, torch.Tensor)
    orig_device = None
    if is_tensor:
        orig_device = iq_data.device
        iq_data = iq_data.cpu().numpy()

    envelope = np.abs(iq_data)
    envelope /= (np.amax(envelope, axis=(-3, -2, -1), keepdims=True) + 1e-12)  # Normalize

    if mode == 'log':
        epsilon = 1e-10
        log_image = 20 * np.log10(envelope + epsilon)
        log_image += dynamic_range_dB
        log_image[log_image < 0] = 0
        result = log_image.astype(np.float32)

    elif mode == 'gamma':
        gamma_image = envelope ** gamma
        gamma_image = np.clip(gamma_image * 255, 0, 255).astype(np.float32)
        result = gamma_image

    else:
        raise ValueError(f"Unsupported compression mode: {mode}. Choose 'log' or 'gamma'.")

    if is_tensor:
        result = torch.tensor(result, dtype=torch.float32, device=orig_device)

    return result


# https://github.com/tristan-deep/dehazing-diffusion/blob/main/processing.py
def mu_law_compress(iq_data, mu=255):
    """Apply μ-law companding to complex I/Q data
    Takes the magnitud and compresses the range of the signal 
    μ detemrines the amount of compression applied"""
    magnitude = torch.abs(iq_data)  #  magnitude
    phase = torch.angle(iq_data)  #  phase

    # Apply μ-law to magnitude (no need of sign because the abs is always positive)
    compressed_mag = torch.log(1 + mu * magnitude) / torch.log(torch.tensor(1 + mu))

    # Reconstruct compressed complex number
    compressed_iq = compressed_mag * torch.exp(1j * phase)
    return compressed_iq

def mu_law_expand(compressed_iq, mu=255):
    """Expand μ-law to compressed I/Q data"""
    compressed_mag = torch.abs(compressed_iq)  # Get magnitude
    phase = torch.angle(compressed_iq)  # Get phase

    # Apply inverse μ-law to magnitude
    expanded_mag = ((1 + mu) ** compressed_mag - 1) / mu

    # Reconstruct expanded complex signal
    expanded_iq = expanded_mag * torch.exp(1j * phase)
    return expanded_iq




# # POSSIBLE FOR ADDING SPECKLE INTO THE TARGET I/Q DATA WHILE KEEPING THE PAHSE AND AMGNITUD
# def wavelet_decompose_3d(volume, wavelet='db4', level=3):
#     return pywt.wavedecn(volume, wavelet=wavelet, level=level)

# def blend_coeffs_3d(conv_coeffs, ref_coeffs, alpha=0.3):
#     blended_coeffs = [conv_coeffs[0]]  # Approximation (low-freq part)
    
#     for conv_detail, ref_detail in zip(conv_coeffs[1:], ref_coeffs[1:]):
#         blended_detail = {}
#         for key in conv_detail.keys():
#             cnn_band = conv_detail[key]
#             ref_band = ref_detail[key]
            
#             # Nonlinear blend
#             blended_band = np.where(
#                 np.abs(ref_band) > np.abs(cnn_band),
#                 (1 - alpha) * cnn_band + alpha * ref_band,
#                 cnn_band
#             )
#             blended_detail[key] = blended_band
        
#         blended_coeffs.append(blended_detail)
    
#     return blended_coeffs

# def wavelet_reconstruct_3d(coeffs, wavelet='db4'):
#     return pywt.waverecn(coeffs, wavelet=wavelet)

# def add_speckle(dwi, conv): 
#     # conv_mag = torch.abs(conv).squeeze(0) # [192,192,192]
#     # conv_phase = torch.angle(conv)
#     # # this works for one input of 9dws and not a batch
#     # dwi_mag = torch.mean(torch.abs(dwi), dim=0) # [192,192,192]
#     conv_mag = np.abs(conv) # [192,192,192]
#     conv_phase = np.angle(conv)
#     # this works for one input of 9dws and not a batch
#     dwi_mag = np.mean(torch.abs(dwi), dim=0) # [192,192,192]

#     # convert to numpy to be able to work on pywave
#     conv_coeff = wavelet_decompose_3d(conv_mag)
#     dwi_coeff = wavelet_decompose_3d(dwi_mag)
#     blended_coeff = blend_coeffs_3d(conv_coeff, dwi_coeff)
#     restored_conv = wavelet_reconstruct_3d(blended_coeff)

#     # convert back to torch
#     real_conv = restored_conv * np.cos(conv_phase)
#     imag_conv = restored_conv * np.sin(conv_phase)
#     complex_result = real_conv + 1j * imag_conv
#     return np.expand_dims(complex_result.astrype(np.complex64), axis=0)
