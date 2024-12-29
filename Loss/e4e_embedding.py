import torch
from torch import nn
import torch.nn.functional as F
from e4e.models.psp import My_pSp
from e4e.models.encoders import psp_encoders
from argparse import Namespace
from torch.autograd import Function

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def embedding_loss(z_id_X, z_id_Y,tags=None):
    l2 = nn.MSELoss()

    if tags == None:
        return l2(z_id_X, z_id_Y).mean()
    else:
        loss = 0
        count =0
        for i in range(min(len(tags),z_id_X.shape[2])):
            if tags[i] == 1:
                loss = loss + l2(z_id_X[:,i,:], z_id_Y[:,i,:]).mean()
                count+=1
        if count == 0:
            return torch.zeros((1)).mean()
        loss = loss / float(count)
        return loss


class SVGL(Function):
    """
    SVGL layer from EXE-GAN
    used to adjust the loss for each pixel.
    it means that pixels on the edges should be not emphasised during training.
    """
    @staticmethod
    def forward(ctx,input,loss_maps):
        """
        :param input: [batch,channel,height,width] (input image)
        :param loss_maps:  [batch,1,height,width] (corresponding weight map)
        :return: input (without any change)
        """
        ctx.save_for_backward(input,loss_maps)

        return input

    @staticmethod
    def backward(ctx,grad_output):
        input,loss_weights = ctx.saved_tensors  # 获取前面保存的参数,也可以使用self.saved_variables

        d_input = grad_output * loss_weights

        return d_input,None

    def ada_piexls(input,loss_maps):
        return SVGL.apply(input,loss_maps)










class E4e_embedding(nn.Module):

    def __init__(self,model_path,out_size,size,device,input_channel=3,use_generator=False):
        """
        :param model_path: ckpt path
        :param out_size: used size e.g. 256,256
        :param size: pre-trained model trained size 1024,1024
        :param device: device cpu or cuda
        :param input_channel:
        :param use_generator: if or not use the generator
        """
        super(E4e_embedding, self).__init__()
        print('Loading E4E ')

        # model_path = 'pre_trained/e4e_ffhq_encode.pt'

        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = model_path
        opts = Namespace(**opts)
        self.E4Enet = My_pSp(opts,out_size,size, device,input_channel=input_channel,use_generator=use_generator).eval().to(device)

    @torch.no_grad()
    def get_w_plus(self,img,weight_map=None):

        if weight_map is not None:
            img = SVGL.ada_piexls(img,weight_map)
        # img = F.interpolate(img,(256,256),mode="bilinear")
        img = F.interpolate(img,(256,256),mode="bilinear")

        w_plus = self.E4Enet(img)
        return w_plus

    @torch.no_grad()
    def get_w_plus_feat(self, img, weight_map=None):

        if weight_map is not None:
            img = SVGL.ada_piexls(img, weight_map)
        img = F.interpolate(img, (256, 256), mode="bilinear")
        w_plus,feats = self.E4Enet.forward_with_feat(img)
        return w_plus, feats

    @torch.no_grad()
    def get_style_mapping(self,styles):
        w_ = self.E4Enet.noise_mapping(styles)
        return  w_

    @torch.no_grad()
    def mean_latent(self, n_latent, style_dim, device):
        latent_in = torch.randn(
            n_latent, style_dim, device=device
        )
        latent = self.E4Enet.style(latent_in).mean(0, keepdim=True)

        return latent


    def get_stylegan_feats_grad(self, styles):
       images,feats = self.E4Enet.stylegan2_feat_forward_with_grad(styles,resize=True, randomize_noise=True,)
       return images,feats


    def get_stylegan_feats(self, styles):
       with torch.no_grad():
           images,feats = self.E4Enet.stylegan2_feat_forward(styles,resize=True, randomize_noise=True,)

       return images,feats



    def get_stylegan_featsV2(self, styles,grad=False,return_feat=True):

        if grad == False:
            with torch.no_grad():
                if return_feat == True:
                    images,feats = self.E4Enet.stylegan2_feat_forward_v2(styles,resize=True, randomize_noise=True,)
                    return images,feats
                else:
                    images  = self.E4Enet.stylegan2_feat_forward_v2(styles,resize=True, randomize_noise=True,return_features=False)
                    return images
        else:
            if return_feat == True:
                images, feats = self.E4Enet.stylegan2_feat_forward_v2(styles, resize=True, randomize_noise=True, )
                return images, feats
            else:
                images = self.E4Enet.stylegan2_feat_forward_v2(styles, resize=True, randomize_noise=True,return_features=False)
                return images



    def open_stylegan_grad(self):
        self.E4Enet.open_decoder_grad()

    def close_stylegan_grad(self):
        self.E4Enet.close_decoder_grad()




