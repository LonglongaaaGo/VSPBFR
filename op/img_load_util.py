import numpy as np
from PIL import Image
import torch
import glob
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import utils


def get_img_lists(path, suffix):
    out = list(glob.glob('{}/*{}'.format(path, suffix)))
    out.sort()
    return out

def load_img2tensor(path,size):
    """
    :param path: a image path
    :param size:  size : 256 ..
    :return: a mask tensor   [1,3,size,size]
    """
    original_image = Image.open(path).convert('RGB')
    original_image = TF.to_tensor(original_image).unsqueeze(0)
    original_image = F.interpolate(original_image, size=(size, size))
    original_image = (original_image - 0.5) * 2
    return original_image

def save_tensor_img(img,save_dir="./",name="_test_"):
    utils.save_image(
        torch.cat([ img]),f"{str(save_dir)}/{name}_.png",
        nrow=int(img.shape[0]),normalize=True,range=(-1, 1))


def load_mask2tensor(path,size):
    """
    :param path: mask path
    :param size:  size : 256 ..
    :return: a mask tensor   [1,1,size,size]
    """
    mask_img = Image.open(path)
    # mask dim and value
    mask_img = np.array(mask_img)
    if mask_img.ndim == 2:
        mask = np.expand_dims(mask_img, axis=0)
    else:
        mask_img = np.transpose(mask_img, (2, 0, 1))
        mask = mask_img[0:1, :, :]
    mask[mask <= 20] = 0
    mask[mask > 20] = 1.0
    masks = torch.from_numpy(mask).unsqueeze(0).float()
    masks = F.interpolate(masks, size=(size, size))

    return masks


if __name__ == '__main__':

    masked_dir = "/home/k/EXE-GAN_cases/Image_Re-composition/mask"
    gt_dir = "/home/k/EXE-GAN_cases/Image_Re-composition/gt_img"
    exemplar_dir = "/home/k/EXE-GAN_cases/Image_Re-composition/exemplar"
    exe_post = "_exemplar.png"
    mask_post = "_mask.png"
    gt_post = "_real.png"

    gt_imgs = get_img_lists(gt_dir,gt_post)
    mask_imgs = get_img_lists(masked_dir,mask_post)

    exe_imgs = get_img_lists(exemplar_dir,exe_post)

    for i in range(len(exe_imgs)):
        exe_img_ = load_img2tensor(exe_imgs[i])
        gt_img_ = load_img2tensor(gt_imgs[i])
        mask_ = load_img2tensor(mask_imgs[i])


