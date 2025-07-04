import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init
from torch.autograd import Function
# from torch.nn.functional import relu, max_pool2d, leaky_relu, prelu
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import complextorch.nn as ctnn
from complextorch.nn.modules.activation.fully_complex import CVSigmoid
from complextorch.nn.modules.activation.split_type_B import CVPolarTanh

    
class ComplexConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.real_conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                                   groups=groups, bias=False)
        self.imag_conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                                   groups=groups, bias=False)

    def forward(self, x): #shape pf x: [batch,channel,d, h, w] dtype = torch.cfloat
        real = self.real_conv(x.real) - self.imag_conv(x.imag)
        imag = self.real_conv(x.imag) + self.imag_conv(x.real)
        return torch.complex(real, imag)

## https://github.com/sweetcocoa/DeepComplexUNetPyTorch/blob/master/DCUNet/complex_nn.py

class ComplexConvTranspose3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.real_convt = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)
        self.imag_convt = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x, output_size=None): #shape pf x: [batch,channel,d, h, w] dtype = torch.cfloat
        real = self.real_convt(x.real) - self.imag_convt(x.imag)
        imag = self.real_convt(x.imag) + self.imag_convt(x.real)

        return torch.complex(real, imag)
    
class ComplexGroupNorm3D(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, track_running_stats=True):
        super().__init__()
        self.real_gn = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)
        self.imag_gn = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)
    
    def forward(self, x):
        real = self.real_gn(x.real)
        imag = self.imag_gn(x.imag)
        return torch.complex(real, imag)


class ComplexBatchNorm3D(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.bn = nn.BatchNorm3d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        return torch.complex(self.bn(x.real), self.bn(x.imag))
    
class ComplexReLU(nn.Module):
    def __init__(self):
        super(ComplexReLU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return torch.complex(self.relu(x.real), self.relu(x.imag))

class ComplexLReLU(nn.Module):
    def __init__(self, negative_slope=0.01, inplace=False):
        super().__init__()
        self.Lrelu = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)

    def forward(self, x):
        return torch.complex(self.Lrelu(x.real), self.Lrelu(x.imag))


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups):
        super(EncoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, dtype=torch.cfloat),
            ComplexBatchNorm3D(out_channels),
            # ComplexGroupNorm3D(num_groups, out_channels),
            ComplexReLU(),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, dtype=torch.cfloat),
            ComplexBatchNorm3D(out_channels),
            # ComplexGroupNorm3D(num_groups, out_channels),
            ComplexReLU()
        )
    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, padding=0, stride=2, dtype=torch.cfloat),
            ComplexBatchNorm3D(out_channels),
            ComplexReLU(),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, dtype=torch.cfloat),
            ComplexBatchNorm3D(out_channels),
            # ComplexGroupNorm3D(num_groups, out_channels),
            ComplexReLU()
        )
    def forward(self, x):
        return self.block(x)
    
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups):
        super(Bottleneck, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, dtype=torch.cfloat),
            ComplexBatchNorm3D(out_channels),
            # ComplexGroupNorm3D(num_groups, out_channels),
            ComplexReLU(),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, dtype=torch.cfloat),
            ComplexBatchNorm3D(out_channels),
            # ComplexGroupNorm3D(num_groups, out_channels),
            ComplexReLU()
        )
    def forward(self, x):
        return self.block(x)

class Unet3Dcx(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder1 = EncoderBlock(9, 16, num_groups=2)
        self.encoder2 = EncoderBlock(16, 32, num_groups=4)
        self.encoder3 = EncoderBlock(32, 64, num_groups=4)

        self.bottleneck = Bottleneck(64, 128, num_groups=4)

        self.decoder1 = DecoderBlock(128 + 64, 64, num_groups=4)
        self.decoder2 = DecoderBlock(64 + 32, 32, num_groups=4)
        self.decoder3 = DecoderBlock(32 + 16, 16, num_groups=2)

        self.final_conv = ComplexConv3D(16, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x) # [B, 16, 96, 96, 96]
        e2 = self.encoder2(e1) # [B, 32, 48, 48, 48]
        e3 = self.encoder3(e2) # [B, 64, 24, 24, 24]

        b = self.bottleneck(e3)  # [B, 128, 24, 24, 24]

        d1 = self.decoder1(torch.cat([b, e3], dim=1))  # [B, 64, 48, 48, 48]
        d2 = self.decoder2(torch.cat([d1, e2], dim=1)) # [B, 32, 96, 96, 96]
        d3 = self.decoder3(torch.cat([d2, e1], dim=1)) # [B, 16, 192, 192, 192]

        return self.final_conv(d3) # [B, 1, 192, 192, 192]


class UNetResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups):
        super(UNetResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, dtype=torch.cfloat),
            ComplexGroupNorm3D(num_groups, out_channels),
            ComplexReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, dtype=torch.cfloat),
            ComplexGroupNorm3D(num_groups, out_channels)
        )

        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2, dtype=torch.cfloat)
            if in_channels != out_channels else nn.Identity()
        )

        self.relu = ComplexReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))
    
class ComplexCBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ComplexCBAM, self).__init__()
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.mlp = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, kernel_size=1, dtype=torch.cfloat),
            ComplexReLU(),
            nn.Conv3d(channels // reduction, channels, kernel_size=1, dtype=torch.cfloat)
        )
        self.sigmoid = CVSigmoid()

        # Spatial attention
        self.conv_spatial = nn.Conv3d(2, 1, kernel_size=7, padding=3, dtype=torch.cfloat)

    def forward(self, x):
        # ----- Channel Attention -----
        avg_out = self.mlp(self.avg_pool(x))           # [B, C, 1, 1, 1] complex
        max_out = self.mlp(self.max_pool(x))           # [B, C, 1, 1, 1] complex
        channel_att = self. sigmoid(avg_out + max_out)       
        # This scales the magnitudes of x channel-wise while preserving phase.    
        x = x * channel_att

        # ----- Spatial Attention -----
        avg = torch.mean(x, dim=1, keepdim=True)         # [B, 1, D, H, W]
        max, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, max], dim=1)  # [B, 2, D, H, W]
        spatial_att = self.sigmoid(self.conv_spatial(x)) # [B, 1, D, H, W] complex
        x = x * spatial_att

        return x
    
class BottleneckWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups):
        super().__init__()
        self.bottleneck = Bottleneck(in_channels, out_channels, num_groups)
        self.attn = ComplexCBAM(out_channels)

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.attn(x)
        return x
    
class CxUnet_RB(nn.Module):
    def __init__(self, in_channels=9, out_channels=1):
        super(CxUnet_RB, self).__init__()

        self.encoder1 = UNetResidualBlock(in_channels, 16, num_groups=2)
        self.encoder2 = UNetResidualBlock(16, 32, num_groups=4)
        self.encoder3 = UNetResidualBlock(32, 64, num_groups=4)

        self.bottleneck = BottleneckWithAttention(64, 128, num_groups=4)

        self.decoder1 = DecoderBlock(128 + 64, 64, num_groups=4)
        self.decoder2 = DecoderBlock(64 + 32, 32, num_groups=4)
        self.decoder3 = DecoderBlock(32 + 16, 16, num_groups=2)

        self.final_conv = nn.Conv3d(16, out_channels, kernel_size=1, dtype=torch.cfloat)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        b = self.bottleneck(e3)

        d1 = self.decoder1(torch.cat([b, e3], dim=1))
        d2 = self.decoder2(torch.cat([d1, e2], dim=1))
        d3 = self.decoder3(torch.cat([d2, e1], dim=1))

        out = self.final_conv(d3)

        return out

# -------------- CID Net in 3D with the amplitude maximization unit (AMU) --------------
class AMU(nn.Module):
    def __init__(self, pieces=4):
        super().__init__()
        self.pieces = pieces

    def forward(self, x):  # x is complex: (B, C, D, H, W)
        x_a = x.abs()
        feature_maps = x_a.shape[1] // self.pieces
        out_shape = (x_a.shape[0], feature_maps, self.pieces, *x_a.shape[2:])
        x_a = x_a.view(out_shape)
        idx = x_a.argmax(dim=2, keepdim=True)
        del x_a

        x = x.view(out_shape)
        x = x.gather(dim=2, index=idx).squeeze(dim=2)
        return x


class CIDNet3D(nn.Module):
    def __init__(self, in_channels = 9, pieces = 4):
        super(CIDNet3D, self).__init__()
        self.pieces = pieces
        self.amu = AMU(pieces=pieces)
        self.layer1 = nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1, dtype=torch.cfloat) #192
        self.layer2 = nn.Conv3d(in_channels=32//pieces, out_channels=64, kernel_size=3, padding=1, dtype=torch.cfloat) #192
        self.layer3 = nn.Conv3d(in_channels=64//pieces, out_channels=32, kernel_size=3, padding=1, dtype=torch.cfloat) #192

        # The multi path with different kernel sizes to get different spatial features
        self.layer4_1 = nn.Conv3d(in_channels=32//pieces, out_channels=4, kernel_size=(5, 7, 7), padding=(2, 3, 3), dtype=torch.cfloat) #192
        self.layer4_2 = nn.Conv3d(in_channels=32//pieces, out_channels=4, kernel_size=(3, 7, 7), padding=(1, 3, 3), dtype=torch.cfloat)
        self.layer4_3 = nn.Conv3d(in_channels=32//pieces, out_channels=4, kernel_size=(1, 3, 3), dtype=torch.cfloat)
        # self.layer4_4 = nn.Conv3d(in_channels=64//pieces, out_channels=4, kernel_size=3, padding=1, dtype=torch.cfloat)
        
        # Final refinement layer
        self.layer5 = nn.Conv3d(in_channels=3, out_channels=4, kernel_size=1, padding=0, dtype=torch.cfloat)

    def forward(self, x):
        x = self.layer1(x)
        x = self.amu(x)

        x = self.layer2(x)
        x = self.amu(x)

        x = self.layer3(x)
        x = self.amu(x)

        x_1 = self.layer4_1(x)
        x_1 = self.amu(x_1)
        x_2 = self.layer4_2(x)
        x_2 = self.amu(x_2)
        x_3 = self.layer4_3(x)
        x_3 = self.amu(x_3)

        x = torch.cat([x_1, x_2, x_3], dim=1)
        del x_1, x_2, x_3 #, x_4
        x = self.layer5(x)
        x = self.amu(x)

        return x


# -------------- Complex RNN with GRU --------------
class ComplexRNN_GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()           # [B, 1, D, H, W] -> [B, F]
        self.gru = ComplexGRU(input_dim=256, hidden_dim=512)
        self.decoder = CNNDecoder()           # [B, H_gru] -> [B, 1, D, H, W]

    def forward(self, x):  # x input shape: [B, C=9, D, H, W]
        # i need to add a dimention to havex: [B, T=9, 1, D, H, W]
        x = x.unsqueeze(2)  # Add a channel dimension: [B, T, C=1, D, H, W]
        B, T, C, D, H, W = x.shape
        # instead of changing the shape i send the data in a loop

        # List to store feature vectors for each time step
        feature_vectors_list = []
        for t in range(T):
            # Extract the t-th time step
            x_t = x[:, t, :, :, :, :] # [B, C, D, H, W] where C=1
            encoded_t = self.encoder(x_t)  # [B, F]
            # print("After encoder:", encoded_t.shape)   
            feature_vectors_list.append(encoded_t)
        
        # Stack the list of feature vectors to form the GRU input sequence [B, T, F] where T=9 and F=256
        gru_input = torch.stack(feature_vectors_list, dim=1) # Stack along dim=1 (sequence length)
        # print("GRU input shape:", gru_input.shape)  # [B, T=9, F=256]
        all_hidden_states, h_n = self.gru(gru_input)               # h_n: [1, B, H_gru] ComplexGRU returns (all_hidden_states, final_h)
        # print("GRU output shape:", h_n.shape)    
        h_n = h_n.squeeze(0)                     # [B, H_gru]
        output = self.decoder(h_n)               # [B, 1, D, H, W]
        return output

class CNNEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1, dtype=torch.cfloat),  # [B, 16, 96, 96, 96]
            ComplexReLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1, dtype=torch.cfloat), # [B, 32, 48, 48, 48]
            ComplexReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1, dtype=torch.cfloat), # [B, 64, 24, 24, 24]
            ComplexReLU(),
            nn.Flatten(start_dim=1),                                          # [B, 64 * 24 * 24 * 24] = [B, 884736]
            nn.Linear(64 * (192//8) * (192//8) * (192//8), feature_dim, dtype=torch.cfloat),          # [B, feature_dim]
            ComplexReLU()
        )

    def forward(self, x):
        return self.encoder(x)
    
class CNNDecoder(nn.Module):
    def __init__(self, feature_dim=512, output_shape=(1, 192, 192, 192)):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 64 * (192//8) * (192//8) * (192//8), dtype=torch.cfloat),  # [B, 64 * H/8 * W/8 * D/8]
            ComplexReLU(),
            # reshape back to the original shape
            nn.Unflatten(1, (64, (192//8),(192//8), (192//8))),  # [B, 64, D/8, H/8, W/8]
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, dtype=torch.cfloat),
            ComplexReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, dtype=torch.cfloat),
            ComplexReLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1, dtype=torch.cfloat),
            # nn.Sigmoid()  # for now no activation function at the end, can be changed later
        )

    def forward(self, x):
        return self.decoder(x)

class ComplexGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.x_z = nn.Linear(input_dim, hidden_dim, dtype=torch.cfloat) # input to hidden
        self.h_z = nn.Linear(hidden_dim, hidden_dim, dtype=torch.cfloat) # hidden to hidden

        self.x_r = nn.Linear(input_dim, hidden_dim, dtype=torch.cfloat)
        self.h_r = nn.Linear(hidden_dim, hidden_dim, dtype=torch.cfloat)

        self.x_h = nn.Linear(input_dim, hidden_dim, dtype=torch.cfloat)
        self.h_h = nn.Linear(hidden_dim, hidden_dim, dtype=torch.cfloat)

        self.sigmoid = CVSigmoid()
        self.tanh = CVPolarTanh()

    def forward(self, x , h):
        # x, h: [B, D] complex-valued

        # Compute update gate
        z = self.sigmoid(self.x_z(x) + self.h_z(h)) #input to hidden + hidden to hidden

        # Compute reset gate
        r = self.sigmoid(self.x_r(x) + self.h_r(h))

        # Compute candidate hidden state
        h_tilde = self.tanh(self.x_h(x) + self.h_h(r * h)) # Element-wise complex multiplication (r * h)

        # Final hidden state update
        h_new = (1 - z) * h + z * h_tilde
        return h_new

class ComplexGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bias=True, batch_first=True):
        super().__init__()
        self.cell = ComplexGRUCell(input_dim, hidden_dim)

    def forward(self, x):
        # x: [B, T, F] complex-valued
        B, T, F = x.shape
        # initialize hidden state it must be [num_layers * num_directions, B, hidden_dim]
        h = torch.zeros(1, B, self.cell.hidden_dim, dtype=torch.cfloat, device=x.device)
        current_h = h[0, :, :] 

        hidden_states = []
        for t in range(T):
            current_x = x[:, t, :] # [B, F]
            current_h = self.cell(current_x, current_h) #update the current hidden state
            hidden_states.append(current_h)
        all_hidden_states = torch.stack(hidden_states, dim=0)  # [T, B, H]
        all_hidden_states = all_hidden_states.transpose(0, 1)  # [B, T, H]
        final_h = current_h.unsqueeze(0) # [1, B, H]
        return all_hidden_states, final_h  # Final complex hidden state

class ComplexRNN_GRU_all_h(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()           # [B, 1, D, H, W] -> [B, F]
        self.gru = ComplexGRU(input_dim=256, hidden_dim=512)
        self.decoder = CNNDecoder()           # [B, H_gru] -> [B, 1, D, H, W]
        self.final_conv = nn.Conv3d(9, 1, kernel_size=1, dtype=torch.cfloat)  # Final convolution to match output shape

    def forward(self, x):  # x input shape: [B, C=9, D, H, W]
        # i need to add a dimention to havex: [B, T=9, 1, D, H, W]
        x = x.unsqueeze(2)  # Add a channel dimension: [B, T, C=1, D, H, W]
        B, T, C, D, H, W = x.shape
        # instead of changing the shape i send the data in a loop

        # List to store feature vectors for each time step
        feature_vectors_list = []
        for t in range(T):
            # Extract the t-th time step
            x_t = x[:, t, :, :, :, :] # [B, C, D, H, W] where C=1
            encoded_t = self.encoder(x_t)  # [B, F]
            # print("After encoder:", encoded_t.shape)   
            feature_vectors_list.append(encoded_t)
        
        # Stack the list of feature vectors to form the GRU input sequence [B, T, F] where T=9 and F=256
        gru_input = torch.stack(feature_vectors_list, dim=1) # Stack along dim=1 (sequence length)
        # print("GRU input shape:", gru_input.shape)  # [B, T=9, F=256]
        all_hidden_states, h_n = self.gru(gru_input)               # h_n: [1, B, H_gru] ComplexGRU returns (all_hidden_states, final_h)
        print("GRU output shape:", h_n.shape)    
        h_n = h_n.squeeze(0)                     # [B, H_gru]
        # i can sent he whole hidden state to the decoder all_hidden_states shape: [B, T, H_gru]
        decoded_vols = [] 
        for t in range(T):
            h_n_t = all_hidden_states[:, t, :] # [B, H_gru]
            decoded_vol = self.decoder(h_n_t)               # [B, 1, D, H, W]
            decoded_vols.append(decoded_vol)
        # Stack the decoded volumes to form the output [B, T, 1, D, H, W]
        output = torch.stack(decoded_vols, dim=1)  # # [B, T=9, 1, D, H, W]
        output = output.squeeze(2)  # Remove the channel dimension: [B, T=9, D, H, W]
        output = self.final_conv(output)
        # output shape: [B, 1, D, H, W]
        return output