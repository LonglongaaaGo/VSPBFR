# -*- coding: UTF-8 -*-
#coding=utf-8
import argparse
import random
import os
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
import my_lpips
from restoration_test import load_ddpm
import datetime
from Loss.e4e_embedding import E4e_embedding

from dataset import ImageFolder_restore_free_form
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
)
from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment
from Loss.id_loss import IDLoss



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


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()

def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


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




def train(args, loader,generator, discriminator, g_optim, d_optim, g_ema, device):
    loader = sample_data(loader)

    save_inter = 500
    show_inter = 2000
    if args.debug == True:
        save_inter = 20
        show_inter = 20

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)


    if args.id_loss_weight>0:
        id_loss = IDLoss(args.arcface_path)

    os.makedirs("./sample",exist_ok=True)
    os.makedirs("./checkpoint",exist_ok=True)

    r1_loss = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)

    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    percept_loss = my_lpips.PerceptualLoss( model="net-lin", net="vgg", use_gpu=device.startswith("cuda"))

    print("args.psp_checkpoint_path",args.psp_checkpoint_path)
    ##init embedding
    psp_embedding = E4e_embedding(args.psp_checkpoint_path, out_size=args.size, size=1024, device=device, input_channel=3, use_generator=True).to(device)

    diffusion = load_ddpm(args.ddpm_ckpt)

    loss_dict["g_percept_loss"] = torch.zeros(1).mean().cuda()

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        low_img,_,real_img = next(loader)
        real_img = real_img.to(device).to(torch.float32)*2.0 -1
        low_img = low_img.to(device).to(torch.float32)*2.0 -1
        #
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        #
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        low_latent = psp_embedding.get_w_plus(low_img)
        infer_embedding = diffusion(x=low_latent, condi_in=low_latent, training=False)

        #
        style_sample, de_feats = psp_embedding.get_stylegan_feats(infer_embedding)
        restored_img = generator(low_img,de_feats,infer_embedding,noise)

        if args.augment:
            #.clone().detach() 保证原tensor不变
            real_img_aug, _ = augment(real_img.clone().detach(), ada_aug_p)
            restored_img_aug, _ = augment(restored_img, ada_aug_p)
        else:
            real_img_aug = real_img
            restored_img_aug = restored_img
        #
        fake_pred = discriminator(restored_img_aug.detach())
        real_pred = discriminator(real_img_aug.detach())
        #
        d_loss = d_logistic_loss(real_pred, fake_pred)
        #
        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()
        #
        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            temp_real_img = real_img.detach().clone()
            temp_real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(temp_real_img, ada_aug_p)
            else:
                real_img_aug = temp_real_img
            real_pred = discriminator(real_img_aug)
            r1_loss = d_r1_loss(real_pred, temp_real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        #train G
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)

        restored_img = generator(low_img,de_feats,infer_embedding,noise)

        if args.augment:
            restored_img_aug, _ = augment(restored_img, ada_aug_p)
        else:
            restored_img_aug = restored_img
        fake_pred = discriminator(restored_img_aug)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        if args.percept_loss_weight>0 :
            g_percept_loss = percept_loss(restored_img, real_img.detach(), weight_map=None).sum() * args.percept_loss_weight
            loss_dict["g_percept_loss"] = g_percept_loss
            g_loss += g_percept_loss

        # id loss
        if args.id_loss_weight > 0:
            g_id_loss = id_loss(restored_img, real_img.detach()) * args.id_loss_weight
            loss_dict["g_id_loss"] = g_id_loss
            g_loss += g_id_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        #
        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)
        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        g_percept_loss_val =  loss_dict["g_percept_loss"].mean().item()
        g_id_loss_val = loss_dict["g_id_loss"].mean().item()


        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}; "
                    f"g_percept_loss_val: {g_percept_loss_val:.4f}"
                    f"g_id_loss_val: {g_id_loss_val:.4f}; "

                )
            )

            if i % show_inter == 0:
                with torch.no_grad():
                    utils.save_image(
                        torch.cat([restored_img_aug,
                                   low_img,
                                   style_sample,
                                   real_img,
                                ]).add(1).mul(0.5),
                        f"sample/{str(i).zfill(6)}_.png",
                        nrow=int(1),
                    )


            if i % save_inter == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                        "iter": i,

                    },
                    f"checkpoint/a_restore_model.pt",
                )

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Visual style prompt trainer")
    parser.add_argument("--path", type=str, help="path to the image dataset, could be the folder")
    parser.add_argument("--iter", type=int, default=500000, help="total training iterations")
    parser.add_argument( "--batch", type=int, default=1, help="batch sizes for each gpu")
    #
    parser.add_argument( "--size", type=int, default=512, help="image sizes for the models")
    parser.add_argument( "--r1", type=float, default=10, help="weight of the r1 regularization")

    parser.add_argument("--d_reg_every", type=int, default=16, help="interval of the applying r1 regularization",)
    parser.add_argument("--g_reg_every",type=int,default=4,help="interval of the applying path length regularization", )
    parser.add_argument("--mixing", type=float, default=0.5, help="probability of latent code mixing" )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--channel_multiplier", type=int,default=2,help="channel multiplier factor for the models. config-f = 2, else = 1",)

    parser.add_argument("--percept_loss_weight", type=float, default=0.5, help="weight of the percept loss" )
    parser.add_argument("--id_loss_weight", type=float, default=0.1, help="weight of the id loss")

    parser.add_argument("--debug",type=bool,default=False,help = "for debugging")

    parser.add_argument( "--local_rank", type=int, default=-1, help="local rank for distributed training")
    parser.add_argument("--augment", action="store_true", help="apply non leaking augmentation")
    parser.add_argument( "--augment_p", type=float, default=0,  help="probability of applying augmentation. 0 = use adaptive augmentation",)
    parser.add_argument("--ada_target", type=float,default=0.6,help="target augmentation probability for adaptive augmentation",)
    parser.add_argument("--ada_length",type=int,default=500 * 1000,help="target duraing to reach augmentation probability for adaptive augmentation",)


    parser.add_argument( "--ckpt",type=str, default=None, help="path to the checkpoints to resume training",)
    parser.add_argument("--ddpm_ckpt", type=str, default="pre-train/code_diffuser.pt", help="ddpm model ckpt path")
    parser.add_argument("--psp_checkpoint_path", type=str, default="pre-train/style_encoder_decoder.pt",help="psp model pretrained model")
    parser.add_argument("--arcface_path", type=str, default="pre-train/Arcface.pth", help="Arcface model pretrained model")
    parser.add_argument("--resume", type=bool,default=False, help="reload => False, resume = > True ",)

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:

        if 'SLURM_LOCALID' in os.environ:  # for slurm scheduler
            #ngpus_per_node 一个节点有几个可用的GPU
            ngpus_per_node = torch.cuda.device_count()
            #local_rank 在一个节点中的第几个进程，local_rank 在各个节点中独立
            args.local_rank = int(os.environ.get("SLURM_LOCALID"))
            #在所有进程中的rank是多少
            args.rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + args.local_rank


            available_gpus = list(os.environ.get('CUDA_VISIBLE_DEVICES').replace(',', ""))

            args.current_device = int(available_gpus[args.local_rank])
            torch.cuda.set_device(args.current_device)

            torch.distributed.init_process_group(backend="nccl", init_method="env://",
            world_size=n_gpu, rank=args.rank,timeout=datetime.timedelta(0, 3600))

        else:
            torch.distributed.init_process_group(backend="nccl", init_method="env://",
                     timeout=datetime.timedelta(0, 3600))
            args.local_rank = torch.distributed.get_rank()
            available_gpus = list(os.environ.get('CUDA_VISIBLE_DEVICES').replace(',', ""))
            args.current_device = int(available_gpus[args.local_rank])

    synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0


    from models.RestoreNet import Restoration_net as Generator
    from models.RestoreNet import Discriminator


    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load models:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            if args.resume:
                args.start_iter = int(ckpt["iter"])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])
        if args.resume == True:
            g_optim.load_state_dict(ckpt["g_optim"])
            d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.current_device],
            output_device=args.current_device,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.current_device],
            output_device=args.current_device,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    dataset = ImageFolder_restore_free_form(root=args.path, transform=transform, im_size=(args.size, args.size))
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )
    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device)

