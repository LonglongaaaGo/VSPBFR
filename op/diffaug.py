# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import torch
import torch.nn.functional as F
from torch.nn import ReflectionPad2d
import random


def DiffAugment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def DiffAugment_withsame_trans(x,y, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
            y = y.permute(0, 3, 1, 2)

        for p in policy.split(','):
            if "translation" in p or "scale" in p:
                for f in AUGMENT_FNS[p]:
                    temp = torch.cat([x,y],1)
                    temp = f(temp)
                    x,y = temp.split([x.shape[1],y.shape[1]],dim=1)
            else:
                for f in AUGMENT_FNS[p]:
                    x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
            y = y.permute(0, 2, 3, 1)

        x = x.contiguous()
        y = y.contiguous()

    return x,y

def DiffAugment_withsame_trans_three(x,y,z, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
            y = y.permute(0, 3, 1, 2)
            z = z.permute(0, 3, 1, 2)

        for p in policy.split(','):
            if "translation" in p  or "scale" in p:
                for f in AUGMENT_FNS[p]:
                    temp = torch.cat([x,y,z],1)
                    temp = f(temp)
                    x,y,z = temp.split([x.shape[1],y.shape[1],z.shape[1]],dim=1)
            else:
                for f in AUGMENT_FNS[p]:
                    x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
            y = y.permute(0, 2, 3, 1)
            z = z.permute(0, 2, 3, 1)

        x = x.contiguous()
        y = y.contiguous()
        z = z.contiguous()


    return x,y,z


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_translation_reflect(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    # x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0],mode="reflect")
    padding = ReflectionPad2d((1, 1, 1, 1))
    x_pad = padding(x)
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_scale(x, ratio=0.125):
    batch_size, channels, height, width = x.size()

    # 随机选择缩放因子
    # scale_factor = torch.rand(batch_size, 1, 1, device=x.device) * ratio + 1.0 - ratio / 2
    scale_factor = random.random() * ratio + 1.0 - ratio / 2

    # 计算缩放后的尺寸
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # 执行缩放操作
    x_scaled = F.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=False)

    # 计算填充和裁剪的参数
    pad_h = height - new_height
    pad_w = width - new_width
    pad_top = torch.randint(0, abs(pad_h) + 1, size=(batch_size,), device=x.device)
    pad_bottom = abs(pad_h) - pad_top
    pad_left = torch.randint(0, abs(pad_w) + 1, size=(batch_size,), device=x.device)
    pad_right = abs(pad_w) - pad_left

    # 执行填充和裁剪操作
    x_padded = F.pad(x_scaled, (pad_left[0].item(), pad_right[0].item(), pad_top[0].item(), pad_bottom[0].item()),mode='reflect')
    b,c,h,w = x_padded.size()
    h_ = (h - height)//2
    w_ = (w - width) //2
    x_cropped = x_padded[:, :, h_:h_+height, w_:w_+width]

    return x_cropped



def rand_scale_05(x, ratio=0.5):
    batch_size, channels, height, width = x.size()

    # 随机选择缩放因子
    # scale_factor = torch.rand(batch_size, 1, 1, device=x.device) * ratio + 1.0 - ratio / 2
    scale_factor = random.random() * ratio + 1.0 - ratio / 2

    # 计算缩放后的尺寸
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # 执行缩放操作
    x_scaled = F.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=False)

    # 计算填充和裁剪的参数
    pad_h = height - new_height
    pad_w = width - new_width
    pad_top = torch.randint(0, abs(pad_h) + 1, size=(batch_size,), device=x.device)
    pad_bottom = abs(pad_h) - pad_top
    pad_left = torch.randint(0, abs(pad_w) + 1, size=(batch_size,), device=x.device)
    pad_right = abs(pad_w) - pad_left

    # 执行填充和裁剪操作
    x_padded = F.pad(x_scaled, (pad_left[0].item(), pad_right[0].item(), pad_top[0].item(), pad_bottom[0].item()),mode='reflect')
    b,c,h,w = x_padded.size()
    h_ = (h - height)//2
    w_ = (w - width) //2
    x_cropped = x_padded[:, :, h_:h_+height, w_:w_+width]

    return x_cropped



def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'reflect_translation': [rand_translation_reflect],
    'dis_scale': [rand_scale],  #控制景深
    'dis_scale_0_5': [rand_scale_05],  # 控制景深
    'cutout': [rand_cutout],
}