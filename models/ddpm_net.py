
import torch.nn as nn
import torch
from ldm.ddpm import DDPM

class ResMLP(nn.Module):
    def __init__(self,n_feats = 512):
        super(ResMLP, self).__init__()
        self.resmlp = nn.Sequential(
            nn.Linear(n_feats , n_feats ),
            nn.LeakyReLU(0.1, True),
        )
    def forward(self, x):
        res=self.resmlp(x)
        return res



class denoise(nn.Module):
    def __init__(self,n_feats = 64, n_denoise_res = 5,timesteps=5):
        super(denoise, self).__init__()
        self.max_period=timesteps*10
        n_featsx4=4*n_feats
        resmlp = [
            nn.Linear(n_featsx4*2+1, n_featsx4),
            nn.LeakyReLU(0.1, True),
        ]
        for _ in range(n_denoise_res):
            resmlp.append(ResMLP(n_featsx4))
        self.resmlp=nn.Sequential(*resmlp)

    def forward(self,x, t,c):
        t=t.float()
        t =t/self.max_period
        t=t.view(-1,1)
        c = torch.cat([c,t,x],dim=1)
        fea = self.resmlp(c)

        return fea

class latent_ddpm:
    def __init__(self,
                 linear_start=0.1,
                 linear_end=0.99,
                 n_denoise_res=1,
                 timesteps=4,
                 ):
        self.denoise = denoise(n_feats=64, n_denoise_res=n_denoise_res, timesteps=timesteps)
        self.diffusion = DDPM(denoise=self.denoise, condition=self.condition, n_feats=64,
                              linear_start=linear_start,linear_end=linear_end, timesteps=timesteps)

    def forward(self,latent,training=False):

        if training:
            IPRS2, pred_IPR_list = self.diffusion(latent)
            return IPRS2, pred_IPR_list
        else:
            IPRS2 = self.diffusion(latent)
            return IPRS2