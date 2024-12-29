import torch
import torch.nn as nn
from torch.nn import Module, LayerNorm
import math
import torch.nn.functional as F

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class spatial_attention(nn.Module):
    """
    去掉了 pooling and  用了 18，512 vs 18，512 的latents
    增加 了 t 的编码
    """
    def __init__(self, in_dim=18,latent_dim=512):
        super(spatial_attention, self).__init__()
        self.in_dim = in_dim
        self.q_matrix = nn.Linear(latent_dim, latent_dim, bias=False)
        self.k_matrix = nn.Linear(latent_dim+1, latent_dim, bias=False)
        # self.k_matrix = nn.Linear(1, in_dim, bias=False)
        self.v_matrix = nn.Linear(latent_dim, latent_dim, bias=False)
        #self.fc = nn.Linear(latent_dim,latent_dim)
        # self.pool = nn.AdaptiveAvgPool2d((1,latent_dim))
        # self.fc = nn.Linear(in_dim, in_dim, bias=False)
        # self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm([latent_dim], elementwise_affine=False)

        self.dk = latent_dim

    def forward(self, w, attribute):
        q = self.q_matrix(w) # B,18,512
        k = self.k_matrix(attribute).permute(0, 2, 1) # B,18,512 ->B,512,18
        v = self.v_matrix(w) # B,18,512
        score = torch.matmul(k,q) / math.sqrt(self.dk)
        # score = torch.matmul(w.transpose(-2, -1), attribute) / math.sqrt(self.dk)
        attention = F.softmax(score, dim=1)
        out = torch.matmul(v,attention)
        #sout = self.fc(out)
        # out = self.pool(out)
        out = self.layer_norm(out)

        return out


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)



class TACC_block(Module):
    """
    把attention_mapping 中的 pooling 去掉 增加更强的表达能力
    增加 step 步长
    """
    def __init__(self, latent_dim=512,in_dim=18):
        super(TACC_block, self).__init__()
        self.pixelnorm = PixelNorm()
        #self.norm2d = LayerNorm([18, latent_dim], elementwise_affine=False)
        self.norm1d = LayerNorm([latent_dim], elementwise_affine=False)
        self.q_matrix = nn.Linear(latent_dim+1, latent_dim, bias=False)
        self.k_matrix = nn.Linear(latent_dim, latent_dim, bias=False)
        self.v_matrix = nn.Linear(latent_dim, latent_dim, bias=False)

        self.gamma_ = nn.Sequential(nn.Linear(latent_dim+1, latent_dim), LayerNorm([latent_dim]), ScaledLeakyReLU(0.2), nn.Linear(latent_dim, latent_dim),nn.Sigmoid())
        self.beta_ = nn.Sequential(nn.Linear(latent_dim+1, latent_dim), LayerNorm([latent_dim]), ScaledLeakyReLU(0.2), nn.Linear(latent_dim, latent_dim),ScaledLeakyReLU(0.2))
        # self.fc = nn.Linear(latent_dim, latent_dim)
        # self.softmax = nn.Softmax(dim=1)
        # self.leakyrelu = LeakyReLU()
        self.attention_layer = spatial_attention(in_dim=in_dim,latent_dim=latent_dim)

        self.dk = 18

    def forward(self, x, embd,step):
        '''
        :param x: B,18,512
        :param embd: B,18,512
        :return:
        '''
        x = self.pixelnorm(x)
        K = self.k_matrix(x)  # B,18,512
        V = self.v_matrix(x)  # B,18,512
        #
        c_embd = torch.cat([embd, step], dim=-1)
        Q = self.q_matrix(c_embd).permute(0, 2, 1)  # B,18,512 -> B,512,18

        score = torch.matmul(K, Q)/ math.sqrt(self.dk)
        # score = self.softmax(score, dim=-1)  # B, 18, 18
        score = F.softmax(score, dim=-1)  #channel self-attention

        h = torch.matmul( score,V)
        # h = score * V  # importance choice
        t = self.attention_layer(x, c_embd)  # translation in layer dim

        h = h + t

        h = self.norm1d(h)
        gamma = self.gamma_(c_embd)
        beta = self.beta_(c_embd)

        h = h * (1.0 + gamma) + beta
        # h = self.leakyrelu(h)

        return h




class Code_diffuser(Module):
    """
    from Dynamic_mapperv2
    对所有latent 计算 self-attention
    # 增加ddpm t的引入
    """
    def __init__(self,timesteps,dim=512):
        super(Code_diffuser, self).__init__()

        self.max_period = timesteps
        self.att_mapper = nn.ModuleList([TACC_block(latent_dim=dim) for _ in range(4)])

    def forward(self, x, embd,t):
        t = t.float()
        t = t / self.max_period
        t = t.view(-1, 1)
        t = t.unsqueeze(-1).repeat([1, embd.shape[1], 1])
        for mapper in self.att_mapper:
            x = mapper(x, embd,t)
        return x






