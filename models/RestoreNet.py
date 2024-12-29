import math
import random
import torch
from torch import nn
from torch.nn import functional as F
from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix

def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class SMART_layer(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        rates = [1,2,4,8],
        activate = True,
    ):
        super().__init__()

        self.rates = rates

        self.ModulatedConv2ds = nn.ModuleList()
        for i, rate in enumerate(rates):
            self.ModulatedConv2ds.append(
                Dilated_ModulatedConv2d(
                    in_channel,
                    out_channel//len(rates),
                    kernel_size,
                    style_dim,
                    upsample=upsample,
                    blur_kernel=blur_kernel,
                    demodulate=demodulate,
                    dilation = rate,
                )
            )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.fusion = ConvLayer(out_channel, out_channel, 3)
        # self.gate = nn.Sequential(
        #     # nn.ReflectionPad2d(1),
        #     nn.Conv2d(in_channel, out_channel, 3, padding=1, dilation=1))

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = None
        if activate:
            self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):

        modulated_style = self.modulation(style)

        out_list = []
        for i in range(len(self.ModulatedConv2ds)):
            out = self.ModulatedConv2ds[i](input,modulated_style)
            out_list.append(out)
        out = torch.cat(out_list,dim=1)
        out = self.fusion(out)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        if self.activate != None:
            out = self.activate(out)

        # mask = self.gate(out)
        # mask = my_layer_norm(mask)
        # mask = torch.sigmoid(mask)
        # return input * (1 - mask) + out * mask
        return out

    def forward_vis(self, input, style, noise=None):
        """
        used to visualize the intemediate tensors
        :param input: 
        :param style: 
        :param noise: 
        :return: 
        """
        modulated_style = self.modulation(style)

        out_list = []
        for i in range(len(self.ModulatedConv2ds)):
            out = self.ModulatedConv2ds[i](input,modulated_style)
            out_list.append(out)
        out = torch.cat(out_list,dim=1)
        out = self.fusion(out)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        if self.activate != None:
            out = self.activate(out)
        out_list.append(out)

        return out,out_list

class Dilated_ModulatedConv2d(nn.Module):
    """
    modulated_style 后的特征直接从外面传进来
    """
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,
            upsample=False,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            fused=True,
            dilation = 1,

    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        self.blur_kernel = blur_kernel
        self.dilation = dilation
        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)*dilation
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        # self.padding = kernel_size // 2
        self.padding = ((kernel_size - 1) * dilation) // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        # self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        """
        :param input: input feature
        :param style: 这个 style 必须是 modualted 版本
        :return:
        """

        batch, in_channel, height, width = input.shape  # 2,512,4,4

        if not self.fused:
            weight = self.scale * self.weight.squeeze(0)
            # style = self.modulation(style)

            if self.demodulate:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

            input = input * style.reshape(batch, in_channel, 1, 1)

            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv2d_gradfix.conv_transpose2d(
                    input, weight, padding=0, stride=2,dilation=self.dilation
                )
                out = self.blur(out)

            elif self.downsample:
                input = self.blur(input)
                out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2,dilation=self.dilation)

            else:
                out = conv2d_gradfix.conv2d(input, weight, padding=self.padding,dilation=self.dilation)

            if self.demodulate:
                out = out * dcoefs.view(batch, -1, 1, 1)

            return out
        # 2,512
        # style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        style = style.view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style  # 2,512,512,3,3

        if self.demodulate:
            # 2,512,512,3,3 == > 2,512 对每个输出通道,进行缩放
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch,dilation=self.dilation
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=0, stride=2, groups=batch,dilation=self.dilation
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding, groups=batch,dilation=self.dilation
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        fused=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        self.blur_kernel = blur_kernel
        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape #2,512,4,4

        if not self.fused:
            weight = self.scale * self.weight.squeeze(0)
            style = self.modulation(style)

            if self.demodulate:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

            input = input * style.reshape(batch, in_channel, 1, 1)

            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv2d_gradfix.conv_transpose2d(
                    input, weight, padding=0, stride=2
                )
                out = self.blur(out)

            elif self.downsample:
                input = self.blur(input)
                out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)

            else:
                out = conv2d_gradfix.conv2d(input, weight, padding=self.padding)

            if self.demodulate:
                out = out * dcoefs.view(batch, -1, 1, 1)

            return out
        #2,512
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style  #2,512,512,3,3

        if self.demodulate:
            #2,512,512,3,3 == > 2,512 对每个输出通道,进行缩放
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise

class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class StyledConv_down(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
            downsample=True
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out



class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out



class Eq_Linear(nn.Module):
    def __init__(self, ch_in, ch_out,lr_mul):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.eq_linear = EqualLinear(ch_in, ch_out, lr_mul=lr_mul, activation="fused_lrelu")

    def forward(self,input):
        b,c,h,w = input.size()
        out = self.pool(input)
        out = self.eq_linear(out.view(b,-1))
        return out


class DilatedEqualConv2d(nn.Module):

    def __init__(
        self, in_channel, out_channel, kernel_size,padding=0, stride=1, dilation = 1,bias=True,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # self.padding = ((kernel_size - 1) * dilation) // 2

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation = self.dilation
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class LargeConvLayer(nn.Sequential):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            bias=True,
            activate=True,
            rates=[1, 2, 4, 8],
    ):
        super().__init__()
        self.downsample = downsample

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        self.dilated_convs = nn.ModuleList()
        for i, rate in enumerate(rates):
            if downsample:
                stride = 2
                self.padding = ((kernel_size - 1) * rate-stride) // 2
            else:
                stride = 1
                self.padding = ((kernel_size - 1) * rate) // 2
            self.dilated_convs.append(
                DilatedEqualConv2d(
                    in_channel,
                    out_channel // len(rates),
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    dilation=rate,
                    bias=bias and not activate,
                )
            )

        self.fusion = ConvLayer(out_channel, out_channel, 1)

        self.activate = None
        if activate:
            self.activate = FusedLeakyReLU(out_channel, bias=bias)

    def forward(self, input):

        if self.downsample:
            input = self.blur(input)

        out_list = []
        for i in range(len(self.dilated_convs)):
            out = self.dilated_convs[i](input)
            out_list.append(out)
        out = torch.cat(out_list, dim=1)
        out = self.fusion(out)
        if self.activate != None:
            out = self.activate(out)

        return out



class Restoration_net(nn.Module):
    """
    新增+  style GAN pre-trained feat 目前只是简单相加
    +  对 encoder 也增加 large style kernel
    """
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size
        self.style_dim = style_dim

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.blur_kernel = blur_kernel
        # self.input = ConstantInput(self.channels[4])
        self.conv1 = SMART_layer(
            self.channels[4], self.channels[4], 3, 4*style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], 4*style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        mlp_size = int(math.log(256, 2))
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        ##
        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                # SMART_layer(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    4*style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                SMART_layer(
                    out_channel, out_channel, 3, 4*style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, 4*style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

        self.encoder_res = [2**i for i in range(int(math.log2(size)), 1, -1)]
        self.encoder(im_size=size,channels=self.channels,nc=4,num_styles= self.n_latent,style_channels=style_dim)


    def encoder(self, im_size,channels, nc,num_styles,style_channels,ndf=32):
        self.down_from_big = LargeConvLayer(3, channels[im_size], kernel_size=1)
        self.log_size = int(math.log(im_size, 2))

        in_channel = channels[im_size]
        self.encoder_convs = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            tmp_channel = channels[2 ** (i)]
            out_channel = channels[2 ** (i - 1)]
            conv1 = SMART_layer(
                in_channel, tmp_channel, 3, 2 * self.style_dim, blur_kernel=self.blur_kernel)

            conv_down = StyledConv_down(tmp_channel, out_channel, 3, 2 * self.style_dim, blur_kernel=self.blur_kernel)

            self.encoder_convs.append(conv1)
            self.encoder_convs.append(conv_down)
            in_channel = out_channel

        #E_4X4
        self.final_layer = LargeConvLayer(in_channel, channels[4],kernel_size=3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4]*4*4, channels[4]*2, activation="fused_lrelu"),
            nn.Dropout2d(0.5),
        )

        self.final_transfer = EqualLinear(channels[4]*2, channels[4] * 4*4, activation="fused_lrelu")


    def encoder_forward(self, imgs,latent,noise):
        batch,c,h,w = imgs.size()

        out = self.down_from_big(imgs)
        features = []

        # jj =0
        for ii in range(0,len(self.encoder_convs),2):
            conv_ = self.encoder_convs[ii]
            out = conv_(out,latent[:, ii],noise[ii])
            features.append(out)
            conv_down = self.encoder_convs[ii+1]
            out = conv_down(out,latent[:, ii],noise[ii+1])
            # jj+=2
            # if (ii - 4)>=0: # 从正数第四个开始，Seblock 每隔四个log2 进行计算
            #     se_block = self.Se_blocks[ii - 4]
            #     out = se_block(features[ii-4],out)
            # features.append(out)
        #batch,channel_4,4,4
        out = self.final_layer(out)
        features.append(out)
        #batch,channels[4]*2
        x_global = self.final_linear(out.view(batch,-1))

        early_layer = self.final_transfer(x_global).view(batch,-1,4,4)
        features[-1] = features[-1] + early_layer

        return x_global,features[::-1]


    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent,device):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)


    def forward(
        self,
        images,
        de_feats,      # styleGAN feats
        pre_styles,    # styleGAN latents
        noise_styles,  # random noise
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        if not input_is_latent:
            noise_styles = [self.style(s) for s in noise_styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)]

        if truncation < 1:
            style_t = []
            for style in noise_styles:
                style_t.append( truncation_latent + truncation * (style - truncation_latent) )
            noise_styles = style_t

        if len(noise_styles) < 2:
            inject_index = self.n_latent

            if noise_styles[0].ndim < 3:
                noise_latent = noise_styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                noise_latent = noise_styles[0]
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            noise_latent = noise_styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            noise_latent2 = noise_styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            noise_latent = torch.cat([noise_latent, noise_latent2], 1)

        pre_latent = pre_styles[:,:noise_latent.shape[1],:]
        latent = torch.cat([pre_latent,noise_latent],dim=-1)

        latent_cp = torch.flip(latent, dims=[1]).clone()
        # noise_cp = torch.flip(noise, dims=[0]).clone()
        noise_cp = noise[::-1]
        x_global,features = self.encoder_forward(images,latent_cp,noise_cp)

        # latent = style_latents
        out = self.conv1(features[0], torch.cat([latent[:, 0],x_global],dim=1), noise=noise[0])
        skip = self.to_rgb1(out,torch.cat([latent[:, 1],x_global],dim=1))

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, torch.cat([latent[:, i],x_global],dim=1), noise=noise1)
            fusion_index = (i + 1) // 2
            feat = features[fusion_index]
            # sty_en_feat = en_feats[fusion_index]
            sty_de_feat = de_feats[fusion_index]  # style
            # out = out + feat + sty_en_feat + sty_de_feat
            out = out + feat  + sty_de_feat
            out = conv2(out, torch.cat([latent[:, i+1],x_global],dim=1), noise=noise2)
            skip = to_rgb(out, torch.cat([latent[:, i+2],x_global],dim=1), skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent
        else:
            return image

    def forward_visualize(
            self,
            images,
            de_feats,  # styleGAN feats
            pre_styles,  # styleGAN latents
            noise_styles,  # random noise
            return_latents=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            noise=None,
            randomize_noise=True,
    ):
        if not input_is_latent:
            noise_styles = [self.style(s) for s in noise_styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)]

        if truncation < 1:
            style_t = []
            for style in noise_styles:
                style_t.append(truncation_latent + truncation * (style - truncation_latent))
            noise_styles = style_t

        if len(noise_styles) < 2:
            inject_index = self.n_latent

            if noise_styles[0].ndim < 3:
                noise_latent = noise_styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                noise_latent = noise_styles[0]
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            noise_latent = noise_styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            noise_latent2 = noise_styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            noise_latent = torch.cat([noise_latent, noise_latent2], 1)

        pre_latent = pre_styles[:, :noise_latent.shape[1], :]
        latent = torch.cat([pre_latent, noise_latent], dim=-1)

        latent_cp = torch.flip(latent, dims=[1]).clone()
        # noise_cp = torch.flip(noise, dims=[0]).clone()
        noise_cp = noise[::-1]
        x_global, features = self.encoder_forward(images, latent_cp, noise_cp)

        out_feat_list = []
        out,out_list = self.conv1.forward_vis(features[0], torch.cat([latent[:, 0], x_global], dim=1), noise=noise[0])
        skip = self.to_rgb1(out, torch.cat([latent[:, 1], x_global], dim=1))

        out_feat_list.append(out_list)
        
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, torch.cat([latent[:, i], x_global], dim=1), noise=noise1)
            fusion_index = (i + 1) // 2
            feat = features[fusion_index]
            # sty_en_feat = en_feats[fusion_index]
            sty_de_feat = de_feats[fusion_index]  # style
            # out = out + feat + sty_en_feat + sty_de_feat
            out = out + feat + sty_de_feat
            out,vis_list = conv2.forward_vis(out, torch.cat([latent[:, i + 1], x_global], dim=1), noise=noise2)
            skip = to_rgb(out, torch.cat([latent[:, i + 2], x_global], dim=1), skip)
            out_feat_list.append(vis_list)

            i += 2

        image = skip

        if return_latents:
            return image, latent,out_feat_list
        else:
            return image,out_feat_list







class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out




class Discriminator(nn.Module):
    def __init__(self, size, input_channel=3,channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.encoder_input_convs = ConvLayer(input_channel, channels[size], 1)

        self.log_size = int(math.log(size, 2))

        in_channel = channels[size]
        self.encoder_convs = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs = ResBlock(in_channel, out_channel, blur_kernel)
            self.encoder_convs.append(convs)
            in_channel = out_channel

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)

        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    # def forward(self, input,ind_y,ind_x,label = "segmentation"):
    def forward(self, input):

        out = self.encoder_input_convs(input)
        for ii in range(len(self.encoder_convs)):
            out = self.encoder_convs[ii](out)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out_feature = self.final_conv(out)

        feature = out_feature.view(batch, -1)
        out = self.final_linear(feature)

        return out


