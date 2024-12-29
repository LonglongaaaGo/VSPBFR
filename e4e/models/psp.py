import matplotlib

matplotlib.use('Agg')
import torch
from torch import nn
from e4e.models.encoders import psp_encoders
from e4e.models.stylegan2.model import Generator
from e4e.configs.paths_config import model_paths
import random
import math


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):

    def __init__(self, opts, device):
        super(pSp, self).__init__()
        self.opts = opts
        self.device = device
        # Define architecture
        self.encoder = self.set_encoder()
        self.decoder = Generator(opts.stylegan_size, 512, 8, channel_multiplier=2)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'Encoder4Editing':
            encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading e4e over the pSp framework from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            self.__load_latent_avg(ckpt)
        else:
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(model_paths['ir_se50'])
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print('Loading decoder weights from pretrained!')
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
            self.__load_latent_avg(ckpt, repeat=self.encoder.style_count)


    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag



class My_pSp(nn.Module):

    def __init__(self, opts,out_size,size, device,input_channel=3,use_generator=False):
        super(My_pSp, self).__init__()
        self.opts = opts
        #input channel
        self.opts.input_channel = input_channel

        self.device = device
        # Define architecture
        self.encoder = self.set_encoder()
        self.style = None
        self.decoder = None
        self.use_generator = use_generator
        if use_generator:
            self.decoder = Generator(opts.stylegan_size, 512, 8, channel_multiplier=2)


        self.face_pool = torch.nn.AdaptiveAvgPool2d((out_size, out_size))
        # Load weights if needed
        self.load_weights()

        requires_grad(self.encoder,False)
        requires_grad(self.style,False)
        if use_generator: requires_grad(self.decoder, False)

        self.log_size = int(math.log(size, 2))
        self.n_latent = self.log_size * 2 - 2

        self.out_log_size = int(math.log(out_size, 2))
        self.out_n_latent = self.out_log_size * 2 - 2
        self.out_size = out_size


    def open_decoder_grad(self):
        if self.decoder!=None: requires_grad(self.decoder, True)

    def close_decoder_grad(self):
        if self.decoder != None: requires_grad(self.decoder, False)

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'Encoder4Editing':
            encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading e4e over the pSp framework from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            #load
            self.decoder = Generator(self.opts.stylegan_size, 512, 8, channel_multiplier=2)
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            self.style = self.decoder.style
            self.__load_latent_avg(ckpt)
            if not self.use_generator:
                del self.decoder
                # del self.decoder
                # self.decoder = None
        else:
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(model_paths['ir_se50'])
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print('Loading decoder weights from pretrained!')
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
            self.__load_latent_avg(ckpt, repeat=self.encoder.style_count)

    def forward(self, x, latent_mask=None, inject_latent=None, alpha=None):

        codes = self.encoder(x)
        # normalize with respect to the center of an average face
        if self.opts.start_from_latent_avg:
            if codes.ndim == 2:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
            else:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        return codes[:,:self.n_latent]

    def forward_with_feat(self, x, latent_mask=None, inject_latent=None, alpha=None):

        codes, feats = self.encoder.forward_with_feat(x)
        # normalize with respect to the center of an average face
        if self.opts.start_from_latent_avg:
            if codes.ndim == 2:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
            else:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        return codes[:,:self.n_latent],feats


    def noise_mapping(self, styles,inject_index=None,truncation=1,
            truncation_latent=None,input_is_latent=False):

        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        return latent

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None



    def stylegan2_feat_forward(self, codes, resize=True, randomize_noise=True,return_features=True ):

        with torch.no_grad():
            images, feats = self.decoder([codes],
                                                 input_is_latent=True,
                                                 randomize_noise=randomize_noise,
                                                 return_features=return_features
                                                 )

        feats = feats[:self.out_n_latent]
        if resize:
            images = self.face_pool(images)

        return images,feats

    def stylegan2_feat_forward_with_grad(self, codes, resize=True, randomize_noise=True,return_features=True ):

        images, feats = self.decoder([codes],
                                             input_is_latent=True,
                                             randomize_noise=randomize_noise,
                                             return_features=return_features
                                             )

        feats = feats[:self.out_n_latent]
        if resize:
            images = self.face_pool(images)

        return images,feats


    def stylegan2_feat_forward_v2(self, codes, resize=True, randomize_noise=True,return_features=True ):

        # return with features
        images, feats = self.decoder([codes],
                                             input_is_latent=True,
                                             randomize_noise=randomize_noise,
                                             return_features=return_features
                                             )

        if resize:
            images = self.face_pool(images)

        if return_features== False:
            return images

        feats = feats[:self.out_n_latent]
        return images,feats


