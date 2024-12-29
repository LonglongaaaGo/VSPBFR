# -*- coding: UTF-8 -*-
#coding=utf-8
import sys
# sys.path.append('./')
# print(sys.path)
import argparse
import random
import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm
import my_lpips

from Loss.id_loss import IDLoss
from dataset import ImageFolder_restore
from distributed_test import get_bigger_batch
from op.utils import set_random_seed
from Loss.e4e_embedding import E4e_embedding
from torch.utils.data import Subset
from ldm.ddpm import My_DDPM as DDPM


from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    # reduce_sum,
    # get_world_size,
)


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


class KDLoss(nn.Module):
    """
    Args:
        loss_weight (float): Loss weight for KD loss. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, temperature=0.15):
        super(KDLoss, self).__init__()

        self.loss_weight = loss_weight
        self.temperature = temperature

    def forward(self, S1_fea, S2_fea):
        """
        Args:
            S1_fea (List): contain shape (N, L) vector.
            S2_fea (List): contain shape (N, L) vector.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        loss_KD_dis = 0
        loss_KD_abs = 0
        for i in range(len(S1_fea)):
            S2_distance = F.log_softmax(S2_fea[i] / self.temperature, dim=1)
            S1_distance = F.softmax(S1_fea[i].detach() / self.temperature, dim=1)
            loss_KD_dis += F.kl_div(
                S2_distance, S1_distance, reduction='batchmean')
            loss_KD_abs += nn.L1Loss()(S2_fea[i], S1_fea[i].detach())
        return self.loss_weight * loss_KD_dis, self.loss_weight * loss_KD_abs



def train(args, loader,test_loader, att_mapper,mapper_optim,device):
    loader = sample_data(loader)

    save_inter = 500
    show_inter = 2000

    if args.debug == True:
        save_inter = 200
        show_inter = 200

    pbar = range(args.iter)
    best_fid =args.best_fid
    best_path=args.best_path

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)


    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
 
    cri_kd= KDLoss()

    percept_loss = my_lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda"))

    if args.id_loss_weight>0:
        id_loss = IDLoss(args.arcface_path)

    data_len = len(test_loader) * args.batch
    print("data_len:%d" % data_len)
    best_evel_batch = get_bigger_batch(data_len, max_num=32)
    print("best_evel_batch:%d" % best_evel_batch)


    loss_dict = {}

    if args.distributed:
        att_module = att_mapper.module

    else:
        att_module = att_mapper

    ##init embedding
    psp_embedding = E4e_embedding(args.psp_checkpoint_path, out_size=args.size, size=1024, device=device,input_channel=3, use_generator=True).to(device)

    denoise = att_mapper
    diffusion = DDPM(denoise=denoise, linear_start=0.1,linear_end=0.99, timesteps=4).to(device)

    os.makedirs("./checkpoint",exist_ok=True)
    os.makedirs("./sample",exist_ok=True)

    # train mapper
    loss_dict["latent_id_loss"] = torch.zeros(1).mean().cuda()
    loss_dict["latent_loss"] = torch.zeros(1).mean().cuda()
    loss_dict["latent_percept_loss"] = torch.zeros(1).mean().cuda()

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break
        low_img, real_img = next(loader)
        real_img = real_img.to(device).to(torch.float32) / 127.5 - 1
        low_img = low_img.to(device).to(torch.float32) * 2.0 - 1

        low_latent = psp_embedding.get_w_plus(low_img)
        target_embedding = psp_embedding.get_w_plus(real_img).detach()
        real_sample = psp_embedding.get_stylegan_featsV2(target_embedding.detach(), grad=False,return_feat=False)  # get gt inversion

        requires_grad(att_mapper, True)
        psp_embedding.open_stylegan_grad()

        pred_latent, pred_IPR_list = diffusion(x=low_latent,condi_in=low_latent,training=True)
        l_kd, l_abs = cri_kd([target_embedding], [pred_IPR_list[-1]])
        latent_loss =  l_abs
        loss_dict["latent_loss"] = latent_loss
        loss_dict["l_kd"] = l_kd

        restore_img = psp_embedding.get_stylegan_featsV2(pred_latent, grad=True, return_feat=False)
        #
        if args.percept_loss_weight > 0:
            latent_percept_loss = percept_loss(restore_img, real_img.detach()).mean() * 0.1
            loss_dict["latent_percept_loss"] = latent_percept_loss
            latent_loss += latent_percept_loss

        if args.id_loss_weight >0 :
            latent_id_loss = id_loss(restore_img,real_img.detach())*0.1
            loss_dict["latent_id_loss"] = latent_id_loss
            latent_loss += latent_id_loss

        att_mapper.zero_grad()
        latent_loss.backward()
        mapper_optim.step()

        torch.cuda.empty_cache()
        psp_embedding.close_stylegan_grad()

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        loss_reduced = reduce_loss_dict(loss_dict)
        latent_loss_val = loss_dict["latent_loss"].mean().item()
        latent_id_loss_val = loss_dict["latent_id_loss"].mean().item()
        latent_percept_loss_val = loss_dict["latent_percept_loss"].mean().item()

        l_kd_loss_val = loss_dict["l_kd"].mean().item()
        if get_rank() == 0:
            pbar.set_description(
                (
                    f"latent_loss_val: {latent_loss_val:.4f}; "
                    f"latent_percept_loss_val: {latent_percept_loss_val:.4f}; "
                    f"l_kd_loss_val: {l_kd_loss_val:.4f}; "
                    f"latent_id_loss_val: {latent_id_loss_val:.4f}; "

                )
            )

            if i % show_inter == 0:
                torch.cuda.empty_cache()
                with torch.no_grad():

                    ori_sample, _ = psp_embedding.get_stylegan_feats(low_latent.detach())
                    refine_sample, _ = psp_embedding.get_stylegan_feats(pred_latent.detach())

                utils.save_image(
                    torch.cat([
                                refine_sample,
                               ori_sample,
                               real_sample,
                                low_img,
                               real_img,
                               ]).add(1).mul(0.5),
                    f"sample/{str(i).zfill(6)}_.png",
                    nrow=int(args.batch),
                )

            if i % save_inter == 0:
                print("saving!!!")
                torch.save(
                    {
                        "att_mapper": att_module.state_dict(),
                        "mapper_optim": mapper_optim.state_dict(),
                        "args": args,
                        "iter": i,
                        "best_path": best_path,
                        "best_fid": best_fid,
                    },
                    f"checkpoint/a_recent_model.pt",)

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="code diffuser trainer")
    parser.add_argument("--path", type=str, help="path to the lmdb dataset")
    parser.add_argument("--iter", type=int, default=200000, help="total training iterations")
    parser.add_argument( "--batch", type=int, default=16, help="batch sizes for each gpu" )
    parser.add_argument("--size", type=int, default=256, help="image sizes for the models")
    parser.add_argument("--g_reg_every",type=int, default=4,help="interval of the applying path length regularization",)
    parser.add_argument("--percept_loss_weight", type=float, default=0.5, help="weight of the percept loss")
    parser.add_argument("--id_loss_weight", type=float, default=0.1, help="weight of the id loss")
    parser.add_argument( "--ckpt", type=str,  default=None, help="path to the checkpoints to resume training",)
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--channel_multiplier",type=int, default=2,  help="channel multiplier factor for the models. config-f = 2, else = 1",)
    parser.add_argument("--debug",type=bool,default=False,help = "for debugging")

    parser.add_argument("--local_rank", type=int, default=-1, help="local rank for distributed training" )

    parser.add_argument("--Tstep",type=int,default=4,help="number of steps",)
    parser.add_argument("--beta1",type=float,default=0.0001,help="beta1",)
    parser.add_argument("--betaT",type=float,default=0.02,help="betaT ",)

    parser.add_argument("--resume", type=bool,default=False, help="reload => False, resume = > True ",)
    parser.add_argument("--logger_path", type=str, default="./logger.txt", help="path to the output the generated images")
    parser.add_argument("--arcface_path", type=str, default="pretrained_model/Arcface.pth", help="Arcface model pretrained model")
    parser.add_argument("--psp_checkpoint_path", type=str, default="Pre-trained/E4E_facial_restore/best_model_33507_13w.pt", help="psp model pretrained model")

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        if args.local_rank != -1:  # for torch.distributed.launch
            args.local_rank = args.local_rank
            args.current_device = args.local_rank
        elif 'SLURM_LOCALID' in os.environ:  # for slurm scheduler
            #ngpus_per_node 一个节点有几个可用的GPU
            ngpus_per_node = torch.cuda.device_count()
            #local_rank 在一个节点中的第几个进程，local_rank 在各个节点中独立
            args.local_rank = int(os.environ.get("SLURM_LOCALID"))
            #在所有进程中的rank是多少
            args.rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + args.local_rank
            available_gpus = list(os.environ.get('CUDA_VISIBLE_DEVICES').replace(',', ""))

            args.current_device = int(available_gpus[args.local_rank])
        import datetime
        torch.cuda.set_device(args.current_device)
        torch.distributed.init_process_group(backend="nccl", init_method="env://",world_size=n_gpu,rank=args.rank,timeout=datetime.timedelta(0,7200))
        synchronize()

    args.latent = 512
    args.n_mlp = 8
    args.start_iter = 0

    from models.CodeDiffuser import Code_diffuser as code_diffuser

    att_mapper = code_diffuser(timesteps=args.Tstep).to(device)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)

    mapper_optim = optim.Adam(
        att_mapper.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )


    set_random_seed(random.randint(0,10000))

    args.best_path = ""
    args.best_fid = 1000

    resume = args.resume
    if args.ckpt is not None:
        print("load models:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            if resume == True:
                args.start_iter = int(ckpt["iter"])
                if "best_path" in ckpt:
                    args.best_path = ckpt["best_path"]
                if "best_fid" in ckpt:
                    args.best_fid = ckpt["best_fid"]

        except ValueError:
            pass
        if resume == True:
            att_mapper.load_state_dict(ckpt["att_mapper"])
            mapper_optim.load_state_dict(ckpt["mapper_optim"])


    if args.distributed:

        att_mapper = nn.parallel.DistributedDataParallel(
            att_mapper,
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

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = ImageFolder_restore(root=args.path, transform=transform, im_size=(args.size, args.size))
    test_data = ImageFolder_restore(root=args.test_path, transform=test_transform, im_size=(args.size, args.size))


    if args.debug== True:
        dataset = Subset(dataset, indices=range(400))
        test_dataset = Subset(test_data, indices=range(400))

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )
    test_loader = data.DataLoader(
        test_data,
        batch_size=args.batch,
        sampler=data_sampler(test_data, shuffle=False, distributed=args.distributed),
        drop_last=True,
    )

    train(args, loader, test_loader,  att_mapper,mapper_optim, device)
