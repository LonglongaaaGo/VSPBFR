# -*- coding: UTF-8 -*-
#coding=utf-8
import argparse


from Logger.Logger import Logger
from Loss.e4e_embedding import E4e_embedding
import os
from op.utils import mkdirs,delete_dirs
import torch
from tqdm import tqdm
from torchvision import transforms, utils
from dataset import ImageFolder_restore_test,ImageFolder_restore_test_no_gt
from torch.utils import data as tudata
from torch.utils import data
import random
from op.utils_train import listdir
from models.CodeDiffuser import Code_diffuser
from ldm.ddpm import My_DDPM as DDPM

try:
    import wandb

except ImportError:
    wandb = None
from distributed import (
    get_rank,

)

def load_ddpm(ddpm_ckpt,device="cuda"):

    ckpt = torch.load(ddpm_ckpt, map_location=lambda storage, loc: storage)

    att_mapper = Code_diffuser(timesteps=4).to(device)
    att_mapper.load_state_dict(ckpt["att_mapper"])
    att_mapper.eval()
    diffusion = DDPM(denoise=att_mapper, linear_start=0.1,linear_end=0.99, timesteps=4).to(device)

    return diffusion


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag




def sample_data(loader):
    while True:
        for batch in loader:
            yield batch



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




def tester_restore_ddpm(args,generator,psp_embedding,diffusion,lq_root,hq_root,eval_dict,data_name,device):

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    if hq_root == "None":
        test_data = ImageFolder_restore_test_no_gt(lq_root=lq_root,transform=test_transform, im_size=(args.size, args.size))
    else:
        test_data = ImageFolder_restore_test(lq_root=lq_root, hq_root=hq_root,
                                             transform=test_transform, im_size=(args.size, args.size))

    test_loader = tudata.DataLoader(
        test_data,
        batch_size=args.batch,
        sampler=data_sampler(test_data, shuffle=False, distributed=args.distributed),
        drop_last=False,
    )

    if get_rank() == 0:
        delete_dirs(eval_dict)
        mkdirs(eval_dict)

    generator.eval()
    print("testing!!! len:%d" % (len(test_loader.dataset)))
    with torch.no_grad():
        for jjj, data in tqdm(enumerate(test_loader)):
            if args.debug == True and jjj > 10: break

            if hq_root == "None":
                low_imgs = data
            else:
                low_imgs, real_imgs = data
                real_imgs = real_imgs.to(device).to(torch.float32)

            low_imgs = low_imgs.to(device).to(torch.float32)
            noise = mixing_noise(low_imgs.shape[0], args.latent, args.mixing, device)
            #
            low_latent = psp_embedding.get_w_plus(low_imgs.detach())
            pre_dic_latent = diffusion(x=low_latent, condi_in=low_latent, training=False)
            style_sample, feats = psp_embedding.get_stylegan_feats(pre_dic_latent)
            restored_img = generator(low_imgs,feats,pre_dic_latent,noise)

            torch.cuda.empty_cache()
            for j, g_img in enumerate(restored_img):
                g_img3 = restored_img[j].squeeze()
                im_in3_ = low_imgs[j].squeeze()

                utils.save_image(
                    g_img3,
                    f"{str(eval_dict)}/{str(jjj * args.batch + j).zfill(6)}_{str(get_rank())}_{data_name}_restore.png",
                    nrow=int(1), normalize=True, range=(-1, 1), )

                utils.save_image(
                    im_in3_,
                    f"{str(eval_dict)}/{str(jjj * args.batch + j).zfill(6)}_{str(get_rank())}_{data_name}_low.png",
                    nrow=int(1), normalize=True, range=(-1, 1), )
                utils.save_image(
                    style_sample[j:j+1],
                    f"{str(eval_dict)}/{str(jjj * args.batch + j).zfill(6)}_{str(get_rank())}_{data_name}_sample.png",
                    nrow=int(1), normalize=True, range=(-1, 1), )

                if hq_root != "None":
                    real_img = real_imgs[j].squeeze()
                    utils.save_image(
                        real_img,
                        f"{str(eval_dict)}/{str(jjj * args.batch + j).zfill(6)}_{str(get_rank())}_{data_name}_gt.png",
                        nrow=int(1), normalize=True, range=(-1, 1), )

    generator.train()
    return  eval_dict


def test_main(args,store_data, g_ema, device):

    logger_list = []
    for i in range(len(store_data)):
        data_name_ = store_data[i]["name"]
        logger_list.append(Logger(path=f"./{data_name_}_test_logger.txt", continue_=True))

    print("args.psp_checkpoint_path",args.psp_checkpoint_path)
    ##init embedding
    psp_embedding = E4e_embedding(args.psp_checkpoint_path, out_size=args.size, size=1024, device=device,
                                  input_channel=3, use_generator=True).to(device)

    for kkkk in range(len(store_data)):
        eval_dict = os.path.join(args.eval_dir, str(i))
        eval_dict = os.path.join(eval_dict, store_data[kkkk]["name"])

        diffusion = load_ddpm(args.ddpm_ckpt, device=device)
        tester_restore_ddpm(args, g_ema, psp_embedding, diffusion, store_data[kkkk]["lq"],
                              store_data[kkkk]["hq"],
                              eval_dict, store_data[kkkk]["name"], device)



def get_store_data(lq_data_str,hq_data_str,name_str):
    """
    convert str to a list of dicts
    :param lq_data_str:
    :param hq_data_str:
    :param name_str:
    :return:
    """
    lq_data_list = str(lq_data_str).strip().split(",")
    hq_data_list = str(hq_data_str).strip().split(",")
    name_list = str(name_str).strip().split(",")
    store_data_list = []

    for i in range(len(lq_data_list)):
        dic_ = {"lq":lq_data_list[i],"hq":hq_data_list[i],"name":name_list[i],}
        store_data_list.append(dic_)

    return store_data_list




if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Visual Style prompt restoration test")
    parser.add_argument( "--batch", type=int, default=1, help="batch sizes for each gpu")
    parser.add_argument("--size", type=int, default=512, help="image sizes for the models")
    parser.add_argument( "--mixing", type=float, default=0.5, help="probability of latent code mixing")
    parser.add_argument("--channel_multiplier", type=int,default=2, help="channel multiplier factor for the models. config-f = 2, else = 1",)

    parser.add_argument("--debug",type=bool,default=False,help = "for debugging")
    parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training",)

    parser.add_argument("--ddpm_ckpt", type=str, default="pre-train/code_diffuser.pt", help="ddpm model ckpt path")
    parser.add_argument("--psp_checkpoint_path", type=str, default="pre-train/style_encoder_decoder.pt", help="psp model pretrained model")
    parser.add_argument("--eval_dir", type=str, default="./eval_dir", help="path to the output the generated images")
    parser.add_argument("--lq_data_list", type=str,default="", help="splitted by , ",)
    parser.add_argument("--hq_data_list", type=str,default="", help="splitted by , ",)
    parser.add_argument("--data_name_list", type=str,default="", help="splitted by , ",)
    parser.add_argument("--ckpt_root", type=str,default="./checkpoint", help="the root for all the ckpt files",)
    args = parser.parse_args()

    args.distributed = False

    args.latent = 512
    args.n_mlp = 8

    from models.RestoreNet import Restoration_net as Generator

    ckpts = []
    listdir(args.ckpt_root,ckpts)

    eval_root = args.eval_dir
    for ckpt_path in ckpts:
        print("======"*30)
        if "pt" != str(ckpt_path).strip().split(".")[-1]: continue
        print("for ckpt test:",ckpt_path)
        args.ckpt = ckpt_path
        g_ema = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(device)

        print(g_ema)
        if args.ckpt is not None:
            print("load models:", args.ckpt)
            try:
                ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
                g_ema.load_state_dict(ckpt["g_ema"])
            except RuntimeError as e:
                print(str(e))
                continue

        name_ = os.path.basename(str(ckpt_path)).strip().split(".")[0]
        args.eval_dir = os.path.join(eval_root, name_)
        g_ema.eval()
        store_data = get_store_data(args.lq_data_list, args.hq_data_list, args.data_name_list)
        test_main(args, store_data, g_ema, device)
        torch.cuda.empty_cache()

