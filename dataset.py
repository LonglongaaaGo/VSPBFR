
from PIL import Image
from torch.utils.data import Dataset
import random
import os
import numpy as np
from op.utils_train import listdir
import cv2
import my_basicsr.my_degradations as degradations
import math
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



class  ImageFolder_restore(Dataset):
    """docstring for ArtDataset"""
    def __init__(self, root, transform=None,im_size=(256,256)):
        """
        :param root:
        :param transform:
        :param im_size:  (h,w)
        """
        super( ImageFolder_restore, self).__init__()
        self.root = root

        self.frame = self._parse_frame()
        self.transform = transform
        self.im_size = im_size

        # #
        self.blur_kernel_size = [19, 20]
        self.kernel_list = ('iso', 'aniso')
        self.kernel_prob= [0.5, 0.5]
        self.blur_sigma= [0.1, 10]
        self.downsample_range= [0.8, 8]
        self.noise_range=[0, 20]
        self.jpeg_range= [60, 100]

        self.color_jitter_prob= None
        self.color_jitter_shift=20
        self.color_jitter_pt_prob= None
        self.gray_prob= None
        self.gt_gray= True

    def _parse_frame(self):
        frame = []
        img_names =[]
        listdir(self.root,img_names)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.JPG' or image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx%len(self.frame)]
        img = Image.open(file).convert('RGB')
        w,h = img.size
        # plt.imshow(img)
        # plt.show()

        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
            new_w = int(ratio * w)
            new_h = int(ratio * h)
            img_scaled = img.resize((new_w,new_h),Image.Resampling.LANCZOS)
            h_rang = new_h - self.im_size[0]
            w_rang = new_w - self.im_size[1]
            h_idx = 0
            w_idx = 0
            if h_rang>0: h_idx = random.randint(0,h_rang)
            if w_rang > 0: w_idx = random.randint(0, w_rang)
            img = img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))
        # img.show()
        # plt.imshow(img)
        # plt.show()

        w, h = self.im_size[1],self.im_size[0]
        img_gt = np.array(img).copy().astype(np.float32)/255.0
        # ------------------------ generate lq image ------------------------ #
        # blur
        assert self.blur_kernel_size[0] < self.blur_kernel_size[1], 'Wrong blur kernel size range'
        cur_kernel_size = random.randint(self.blur_kernel_size[0], self.blur_kernel_size[1]) * 2 + 1
        kernel = degradations.random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            cur_kernel_size,
            self.blur_sigma,
            self.blur_sigma, [-math.pi, math.pi],
            noise_range=None)
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if self.noise_range is not None:
            img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

        # resize to original size
        img_lq = cv2.resize(img_lq, (self.im_size[1], self.im_size[0]), interpolation=cv2.INTER_LINEAR)

        # random color jitter (only for lq)
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_lq = self.color_jitter(img_lq, self.color_jitter_shift)
        # random to gray (only for lq)
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
            img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
            if self.gt_gray:
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
                img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])

        # BGR to RGB, HWC to CHW, numpy to tensor
        # img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # if self.transform:
        #     img = self.transform(img)
        img_lq = np.clip((img_lq * 255.0).round(), 0, 255) / 255.

        img_gt = np.array(img)
        img_gt = np.ascontiguousarray(img_gt.transpose(2, 0, 1))  # HWC => CHW
        img_lq = np.ascontiguousarray(img_lq.transpose(2, 0, 1))  # HWC => CHW

        return img_lq,img_gt


    def degrade_img(self,img):


        w,h = img.size
        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
            new_w = int(ratio * w)
            new_h = int(ratio * h)
            img_scaled = img.resize((new_w,new_h),Image.Resampling.LANCZOS)
            h_rang = new_h - self.im_size[0]
            w_rang = new_w - self.im_size[1]
            h_idx = 0
            w_idx = 0
            if h_rang>0: h_idx = random.randint(0,h_rang)
            if w_rang > 0: w_idx = random.randint(0, w_rang)
            img = img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))

        w, h = self.im_size[1], self.im_size[0]
        img_gt = np.array(img).copy().astype(np.float32) / 255.0
        # ------------------------ generate lq image ------------------------ #
        # blur
        assert self.blur_kernel_size[0] < self.blur_kernel_size[1], 'Wrong blur kernel size range'
        cur_kernel_size = random.randint(self.blur_kernel_size[0], self.blur_kernel_size[1]) * 2 + 1
        kernel = degradations.random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            cur_kernel_size,
            self.blur_sigma,
            self.blur_sigma, [-math.pi, math.pi],
            noise_range=None)
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if self.noise_range is not None:
            img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

        # resize to original size
        img_lq = cv2.resize(img_lq, (self.im_size[1], self.im_size[0]), interpolation=cv2.INTER_LINEAR)

        # random color jitter (only for lq)
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_lq = self.color_jitter(img_lq, self.color_jitter_shift)
        # random to gray (only for lq)
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
            img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
            if self.gt_gray:
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
                img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])

        # BGR to RGB, HWC to CHW, numpy to tensor
        # img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # if self.transform:
        #     img = self.transform(img)
        img_lq = np.clip((img_lq * 255.0).round(), 0, 255) / 255.

        img_gt = np.array(img)
        img_gt = np.ascontiguousarray(img_gt.transpose(2, 0, 1))  # HWC => CHW
        img_lq = np.ascontiguousarray(img_lq.transpose(2, 0, 1))  # HWC => CHW

        return img_lq, img_gt




class  ImageFolder_restore_free_form(Dataset):
    """docstring for ArtDataset"""
    def __init__(self, root, transform=None,im_size=(256,256)):
        """
        :param root:
        :param transform:
        :param im_size:  (h,w)
        """
        super( ImageFolder_restore_free_form, self).__init__()
        self.root = root

        self.frame = self._parse_frame()
        self.transform = transform
        self.im_size = im_size

        # #
        self.blur_kernel_size = [19, 20] #ok
        self.kernel_list = ('iso', 'aniso') #ok
        self.kernel_prob= [0.5, 0.5]  #ok
        self.blur_sigma= [0.1, 10]  #ok
        self.downsample_range= [0.8, 8]  #ok
        self.noise_range=[0, 20]  #ok
        self.jpeg_range= [60, 100]  #ok

        self.color_jitter_prob= None  #ok
        self.color_jitter_shift=20  #ok
        self.color_jitter_pt_prob= None  #ok
        self.gray_prob= 0.008  #ok
        self.gt_gray= True  #ok
        self.hazy_prob = 0.008  #ok
        self.hazy_alpha = [0.75, 0.95]  #ok

        # self.shift_prob =  0.2  #ok
        self.shift_prob =  0  #ok  #暂时去掉，
        self.shift_unit = 1  #ok
        self.shift_max_num = 32  #ok

    def _parse_frame(self):
        frame = []
        img_names =[]
        listdir(self.root,img_names)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(self.root, img_names[i])
            if image_path[-4:] == '.JPG' or image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        file = self.frame[idx]
        img = Image.open(file).convert('RGB')
        w,h = img.size
        # plt.imshow(img)
        # plt.show()

        # RandomHorizontalFlip
        flip = random.randint(0, 1)
        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
            new_w = int(ratio * w)
            new_h = int(ratio * h)
            img_scaled = img.resize((new_w,new_h),Image.Resampling.LANCZOS)
            h_rang = new_h - self.im_size[0]
            w_rang = new_w - self.im_size[1]
            h_idx = 0
            w_idx = 0
            if h_rang>0: h_idx = random.randint(0,h_rang)
            if w_rang > 0: w_idx = random.randint(0, w_rang)
            img = img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))
        # img.show()
        # plt.imshow(img)
        # plt.show()

        w, h = self.im_size[1],self.im_size[0]
        img_gt = np.array(img).copy().astype(np.float32)/255.0

        if (self.shift_prob is not None) and (np.random.uniform() < self.shift_prob):
            # self.shift_unit = 32
            # import pdb
            # pdb.set_trace()
            shift_vertical_num = np.random.randint(0, self.shift_max_num * 2 + 1)
            shift_horisontal_num = np.random.randint(0, self.shift_max_num * 2 + 1)
            shift_v = self.shift_unit * shift_vertical_num
            shift_h = self.shift_unit * shift_horisontal_num
            img_gt_pad = np.pad(img_gt, ((self.shift_max_num * self.shift_unit, self.shift_max_num * self.shift_unit),
                                         (self.shift_max_num * self.shift_unit, self.shift_max_num * self.shift_unit),
                                         (0, 0)),
                                mode='symmetric')
            img_gt = img_gt_pad[shift_v:shift_v + h, shift_h: shift_h + w, :]

        img_lq1 = self.degrade_img(img_gt)
        img_lq2 = self.degrade_img(img_gt)

        # random to gray (only for lq)
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            img_lq1 = cv2.cvtColor(img_lq1, cv2.COLOR_BGR2GRAY)
            img_lq2 = cv2.cvtColor(img_lq2, cv2.COLOR_BGR2GRAY)

            img_lq1 = np.tile(img_lq1[:, :, None], [1, 1, 3])
            img_lq2 = np.tile(img_lq2[:, :, None], [1, 1, 3])

            if self.gt_gray:
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
                img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])


        # img_gt = np.array(img)

        img_gt = np.ascontiguousarray(img_gt.transpose(2, 0, 1))  # HWC => CHW
        img_lq1 = np.ascontiguousarray(img_lq1.transpose(2, 0, 1))  # HWC => CHW
        img_lq2 = np.ascontiguousarray(img_lq2.transpose(2, 0, 1))  # HWC => CHW

        return img_lq1,img_lq2,img_gt


    def degrade_img(self,img_gt):
        w, h = self.im_size[1],self.im_size[0]

        # ------------------------ generate lq image ------------------------ #
        # blur
        assert self.blur_kernel_size[0] < self.blur_kernel_size[1], 'Wrong blur kernel size range'
        cur_kernel_size = random.randint(self.blur_kernel_size[0], self.blur_kernel_size[1]) * 2 + 1
        kernel = degradations.random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            cur_kernel_size,
            self.blur_sigma,
            self.blur_sigma, [-math.pi, math.pi],
            noise_range=None)
        img_lq = cv2.filter2D(img_gt, -1, kernel)

        ## add simple hazy
        if (self.hazy_prob is not None) and (np.random.uniform() < self.hazy_prob):
            alpha = np.random.uniform(self.hazy_alpha[0], self.hazy_alpha[1])
            img_lq = img_lq * alpha + np.ones_like(img_lq) * (1 - alpha)

        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if self.noise_range is not None:
            img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

        # resize to original size
        img_lq = cv2.resize(img_lq, (self.im_size[1], self.im_size[0]), interpolation=cv2.INTER_LINEAR)

        # random color jitter (only for lq)
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_lq = self.color_jitter(img_lq, self.color_jitter_shift)


        # BGR to RGB, HWC to CHW, numpy to tensor
        # img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # if self.transform:
        #     img = self.transform(img)
        img_lq = np.clip((img_lq * 255.0).round(), 0, 255) / 255.
        return img_lq



class  ImageFolder_restore_test(Dataset):
    """docstring for ArtDataset"""
    def __init__(self, lq_root,hq_root, transform=None,im_size=(256,256)):
        """
        :param root:
        :param transform:
        :param im_size:  (h,w)
        """
        super( ImageFolder_restore_test, self).__init__()
        self.lq_root = lq_root
        self.hq_root = hq_root

        self.lq_frame = self._parse_frame(lq_root)
        self.hq_frame = self._parse_frame(hq_root)

        self.transform = transform
        self.im_size = im_size

    def _parse_frame(self,root):
        frame = []
        img_names=[]
        listdir(root,img_names)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(root, img_names[i])
            if image_path[-4:] == '.JPG' or image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.lq_frame)

    def __getitem__(self, idx):
        lq_file = self.lq_frame[idx]
        lq_img = Image.open(lq_file).convert('RGB')
        hq_file = self.hq_frame[idx]
        hq_img = Image.open(hq_file).convert('RGB')
        w,h = hq_img.size

        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
            new_w = int(ratio * w)
            new_h = int(ratio * h)
            lq_img_scaled = lq_img.resize((new_w,new_h),Image.Resampling.LANCZOS)
            hq_img_scaled = hq_img.resize((new_w,new_h),Image.Resampling.LANCZOS)

            h_rang = new_h - self.im_size[0]
            w_rang = new_w - self.im_size[1]
            h_idx = 0
            w_idx = 0
            if h_rang>0: h_idx = h_rang//2
            if w_rang > 0: w_idx = w_rang//2
            lq_img = lq_img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))
            hq_img = hq_img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))

        if self.transform:
            lq_img = self.transform(lq_img)
            hq_img = self.transform(hq_img)

        return lq_img,hq_img




class  ImageFolder_restore_test_no_gt(Dataset):
    """docstring for ArtDataset"""
    def __init__(self, lq_root, transform=None,im_size=(256,256)):
        """
        :param root:
        :param transform:
        :param im_size:  (h,w)
        """
        super( ImageFolder_restore_test_no_gt, self).__init__()
        self.lq_root = lq_root

        self.lq_frame = self._parse_frame(lq_root)

        self.transform = transform
        self.im_size = im_size

    def _parse_frame(self,root):
        frame = []
        img_names=[]
        listdir(root,img_names)
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(root, img_names[i])
            if image_path[-4:] == '.JPG' or image_path[-4:] == '.jpg' or image_path[-4:] == '.png' or image_path[-5:] == '.jpeg':
                frame.append(image_path)
        return frame

    def __len__(self):
        return len(self.lq_frame)

    def __getitem__(self, idx):
        lq_file = self.lq_frame[idx]
        lq_img = Image.open(lq_file).convert('RGB')

        w,h = lq_img.size

        if h != self.im_size[0] or w != self.im_size[1]:
            ratio = max(1.0 * self.im_size[0] / h, 1.0 * self.im_size[1] / w)
            new_w = int(ratio * w)
            new_h = int(ratio * h)
            lq_img_scaled = lq_img.resize((new_w,new_h),Image.Resampling.LANCZOS)

            h_rang = new_h - self.im_size[0]
            w_rang = new_w - self.im_size[1]
            h_idx = 0
            w_idx = 0
            if h_rang>0: h_idx = h_rang//2
            if w_rang > 0: w_idx = w_rang//2
            lq_img = lq_img_scaled.crop((w_idx,h_idx,int(w_idx+self.im_size[1]),int(h_idx+self.im_size[0])))

        if self.transform:
            lq_img = self.transform(lq_img)

        return lq_img




def dilate_demo(d_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义结构元素的形状和大小
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # 椭圆形
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (6, 6))  # 十字形
    image = cv2.dilate(d_image, kernel)  # 膨胀操作
    # plt_show_Image_image(image)
    return image


def erode_demo(e_image):
    kernel_size = random.randint(3,7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))  # 定义结构元素的形状和大小  矩形
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # 椭圆形
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (6, 6))  # 十字形
    image = cv2.erode(e_image, kernel)  # 腐蚀操作
    # plt_show_Image_image(image)
    return image
    # 腐蚀主要就是调用cv2.erode(img,kernel,iterations)，这个函数的参数是
    # 第一个参数：img指需要腐蚀的图
    # 第二个参数：kernel指腐蚀操作的内核，默认是一个简单的3X3矩阵，我们也可以利用getStructuringElement（）函数指明它的形状
    # 第三个参数：iterations指的是腐蚀次数，省略是默认为1
