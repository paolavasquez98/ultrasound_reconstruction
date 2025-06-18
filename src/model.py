import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init
from torch.autograd import Function
# from torch.nn.functional import relu, max_pool2d, leaky_relu, prelu
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import complextorch.nn as ctnn
from complextorch.nn import functional as cvF
from complextorch.nn.modules.pooling import AdaptiveAvgPool3d
# from complextorch.nn.modules.activation.split_type_B import modReLU
    
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
            ComplexConv3D(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
            # ComplexBatchNorm3D(out_channels),
            ComplexGroupNorm3D(num_groups, out_channels),
            # cplx_nn.CVBatchNorm3d(out_channels),
            ComplexReLU(),
            # modReLU(bias=-1.0),

            ComplexConv3D(out_channels, out_channels, kernel_size=3, padding=1),
            # ComplexBatchNorm3D(out_channels),
            ComplexGroupNorm3D(num_groups, out_channels),
            # cplx_nn.CVBatchNorm3d(out_channels),
            ComplexReLU()
            # modReLU(bias=-1.0),
        )
    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            ComplexConvTranspose3D(in_channels, out_channels, kernel_size=2, padding=0, stride=2),
            # ComplexBatchNorm3D(out_channels),
            ComplexGroupNorm3D(num_groups, out_channels),
            # cplx_nn.CVBatchNorm3d(out_channels),
            ComplexReLU(),
            # modReLU(bias=-1.0),

            ComplexConv3D(out_channels, out_channels, kernel_size=3, padding=1),
            # ComplexBatchNorm3D(out_channels),
            ComplexGroupNorm3D(num_groups, out_channels),
            # cplx_nn.CVBatchNorm3d(out_channels),
            ComplexReLU()
            # modReLU(bias=-1.0),
        )
    def forward(self, x):
        return self.block(x)
    
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups):
        super(Bottleneck, self).__init__()
        self.block = nn.Sequential(
            ComplexConv3D(in_channels, out_channels, kernel_size=3, padding=1),
            # ComplexBatchNorm3D(out_channels),
            ComplexGroupNorm3D(num_groups, out_channels),
            # cplx_nn.CVBatchNorm3d(out_channels),
            ComplexReLU(),
            # modReLU(bias=-1.0),

            ComplexConv3D(out_channels, out_channels, kernel_size=3, padding=1),
            # ComplexBatchNorm3D(out_channels),
            ComplexGroupNorm3D(num_groups, out_channels),
            # cplx_nn.CVBatchNorm3d(out_channels),
            ComplexReLU()
            # modReLU(bias=-1.0),
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
        self.avg_pool = ctnn.modules.pooling.AdaptiveAvgPool3d(1)

        self.mlp = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, kernel_size=1, dtype=torch.cfloat),
            ComplexReLU(),
            nn.Conv3d(channels // reduction, channels, kernel_size=1, dtype=torch.cfloat)
        )

        # Spatial attention
        self.conv_spatial = nn.Conv3d(1, 1, kernel_size=7, padding=3, dtype=torch.cfloat)

    def forward(self, x):
        # ----- Channel Attention -----
        avg_out = self.avg_pool(x)            # shape: [B, C, 1, 1, 1]
        channel_att = torch.sigmoid(self.mlp(avg_out))
        x = x * channel_att

        # ----- Spatial Attention -----
        avg = torch.mean(x, dim=1, keepdim=True)         # [B, 1, D, H, W]
        spatial_att = torch.sigmoid(self.conv_spatial(avg))
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

        # The muklti path with different kernel sizes to get different spatial features
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


def init_complex_conv3d_weight(kernel_size, input_dim, output_dim, init_mode='HeInit', device='cpu'):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    elif isinstance(kernel_size, tuple) and len(kernel_size) != 3:
        raise ValueError("kernel_size must be an int or a tuple of 3 integers for 3D convolution.")
    
    n_row = output_dim * input_dim
    n_col = kernel_size[0] * kernel_size[1] * kernel_size[2]

    r = torch.randn(n_row, n_col, device=device)
    i = torch.randn(n_row, n_col, device=device)
    z = r + 1j * i

    # SVD-based unitary initialization
    u, _, v = torch.linalg.svd(z, full_matrices=False)
    z_unitary = u @ torch.conj(v)

    real_unitary = z_unitary.real
    imag_unitary = z_unitary.imag

    real_reshape = real_unitary.reshape(output_dim, input_dim, *kernel_size)
    imag_reshape = imag_unitary.reshape(output_dim, input_dim, *kernel_size)

    if init_mode == 'HeInit':
        desired_var = 1. / input_dim
    elif init_mode == 'GlorotInit':
        desired_var = 1. / (input_dim + output_dim)
    else:
        raise ValueError(f"Unsupported init_mode '{init_mode}'")

    real_weight = real_reshape * torch.sqrt(desired_var / torch.var(real_reshape))
    imag_weight = imag_reshape * torch.sqrt(desired_var / torch.var(imag_reshape))

    return real_weight, imag_weight

def apply_complex_init(module):
    if isinstance(module, ComplexConv3D):
        real_w, imag_w = init_complex_conv3d_weight(
            kernel_size=module.real_conv.kernel_size,
            input_dim=module.real_conv.in_channels,
            output_dim=module.real_conv.out_channels,
            init_mode='HeInit',
            device=module.real_conv.weight.device
        )
        with torch.no_grad():
            module.real_conv.weight.copy_(real_w)
            module.imag_conv.weight.copy_(imag_w)

            # Initialize biases to zero (or custom value)
            if module.real_conv.bias is not None:
                module.real_conv.bias.zero_()
                module.imag_conv.bias.zero_()

    elif isinstance(module, ComplexConvTranspose3D):
        real_w, imag_w = init_complex_conv3d_weight(
            kernel_size=module.real_convt.kernel_size,
            input_dim=module.real_convt.out_channels,  # note: swapped
            output_dim=module.real_convt.in_channels,
            init_mode='HeInit',
            device=module.real_convt.weight.device
        )
        with torch.no_grad():
            module.real_convt.weight.copy_(real_w)
            module.imag_convt.weight.copy_(imag_w)

            if module.real_convt.bias is not None:
                module.real_convt.bias.zero_()
                module.imag_convt.bias.zero_()

# ------------other options for the encoder and decoder blocks----------------
class EncoderBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ComplexConv3D(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
            ComplexBatchNorm3D(out_channels),
            ComplexLReLU(),

            ComplexConv3D(out_channels, out_channels, kernel_size=3, padding=1),
            ComplexBatchNorm3D(out_channels),
            ComplexLReLU(),
        )
    def forward(self, x):
        return self.block(x)

class DecoderBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ComplexConvTranspose3D(in_channels, out_channels, kernel_size=2, padding=0, stride=2),
            ComplexBatchNorm3D(out_channels),
            ComplexLReLU(),

            ComplexConv3D(out_channels, out_channels, kernel_size=3, padding=1),
            ComplexBatchNorm3D(out_channels),
            ComplexLReLU(),
        )
    def forward(self, x):
        return self.block(x)
    
class Bottleneck2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ComplexConv3D(in_channels, out_channels, kernel_size=3, padding=1),
            ComplexBatchNorm3D(out_channels),
            ComplexLReLU(),

            ComplexConv3D(out_channels, out_channels, kernel_size=3, padding=1),
            ComplexBatchNorm3D(out_channels),
            ComplexLReLU(),
        )
    def forward(self, x):
        return self.block(x)

class Unet3Dcx2(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder1 = EncoderBlock2(9, 16)
        self.encoder2 = EncoderBlock2(16, 32)
        self.encoder3 = EncoderBlock2(32, 64)

        self.bottleneck = Bottleneck2(64, 128)

        self.decoder1 = DecoderBlock2(128 + 64, 64)
        self.decoder2 = DecoderBlock2(64 + 32, 32)
        self.decoder3 = DecoderBlock2(32 + 16, 16)

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