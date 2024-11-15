import numpy as np
from modules import *
from utils import complex_sig, pwr_normalize



class Relay_Full_Duplex(nn.Module):
    # A template for the following models; capable for CNN, RNN and transformers
    def __init__(self,  args, enc, dec, relay_enc):
        super().__init__()

        # CNN
        self.c_feat = args.cfeat
        self.c_out = args.cout

        # Transformer
        self.n_patches = args.n_patches                 # total # of patches
        self.n_feat = args.n_feat                       # # of features/patch
        self.hidden_size = args.hidden_size             # transformer dimension
        self.n_heads = args.n_heads                     # multihead attention
        self.n_layers = args.n_layers                   # # of transformer layers

        self.n_part_dim = int(np.sqrt(self.n_patches))  # the partition is same for 2 dimensions
        
        self.unit_trans_feat = args.unit_trans_feat     # bandwidth unit 
        self.max_trans_feat = args.max_trans_feat       # num of transmit feat/patch

        # Relay settings
        self.Ps = 10**(args.P/10)
        self.Pr = 10**(args.P/10)    # set them to be the same
        self.P_rng = args.P_rng
        self.gamma1 = args.gamma1
        self.gamma2 = args.gamma2
        self.gamma_rng = args.gamma_rng

        self.layers = args.layers
        self.layer_rng = args.layer_rng
        self.unit = args.unit

        self.unit_sym = int(self.unit_trans_feat*self.n_patches/2)
        self.args = args

        self.enc = enc                      # Source encoder
        self.relay_enc = relay_enc          # generate parity symbols
        self.dec = dec                      # Source decoder
        self.adapt = args.adapt
        self.device = args.device
    
    def gen_transformer_mask(self, latent_shape, a):
        # mask for the transformer -- successively transmit features
        mask = torch.zeros(latent_shape).to(self.device)   # (B, n_patches, (self.layers+1)*self.unit_trans_feat)
        mask[:, :, 0:a*self.unit_trans_feat] = 1
        return mask

    def gen_resnet_mask(self, latent_shape, a):
        # mask for the ResNet
        mask = torch.zeros(latent_shape).to(self.device)   # (B, (self.layers+1)*self.unit_trans_feat, n_feat)
        mask[:, 0:a*self.unit_trans_feat, :, :] = 1
        return mask


    def gen_channel_coeff(self, is_train = False):

        if self.args.adapt and is_train:
            gamma1 = self.gamma1 + self.gamma_rng*(2*torch.rand(1)-1).to(self.device)
            gamma2 = self.gamma2 + self.gamma_rng*(2*torch.rand(1)-1).to(self.device)
        else:
            gamma1 = self.gamma1 + self.gamma_rng*torch.tensor([0]).to(self.device)
            gamma2 = self.gamma2 + self.gamma_rng*torch.tensor([0]).to(self.device)
            
        attn_items = torch.cat((gamma1, gamma2)).unsqueeze(0).unsqueeze(0)                     # [1,1,3]
        g_sr = 10**(gamma1/10)
        g_rd = 10**(gamma2/10)

        return attn_items, g_sr, g_rd


    def relay_pwnorm(self, x):
        x_mean = torch.mean(x)
        x_std = torch.std(x)
        norm_x = (x-x_mean)/x_std
        return norm_x

    def relay_pwnorm_af(self, x):
        x_std = torch.std(x)
        norm_x = x/x_std
        return norm_x
    
    def _to_rnn_input(self, x):
        # x: (B, -1); complex
        seq_x = x.view(x.shape[0], self.layers+1, self.unit_sym)                    # complex
        seq_x = torch.view_as_real(seq_x).view(seq_x.shape[0], self.layers+1, -1)   # (B, layers+1, num_sym); real

        return seq_x





class RelayPFRD_Transformer_Adapt(Relay_Full_Duplex):
    def __init__(self,  args, enc, source_dec, relay_enc):
        super().__init__(args, enc, source_dec, relay_enc)

        self.layers = args.layers+1


        self.register_buffer('x_mean', torch.zeros(1))
        self.register_buffer('x_std', torch.zeros(1))

    def full_duplex_relaying(self, x, attn_item = None):
        # generate the 'parity' block-wise
        
        B, _, total_trans_feat = x.shape                           # （B, n_patches, layers*unit_trans_feat）
        relay_code = torch.zeros((B, self.n_patches, self.layers, self.unit_trans_feat)).to(x.device)

        # transformer mask
        for l in range(self.layers - 1):
            mask_l = self.gen_transformer_mask((B, self.n_patches, total_trans_feat), l+1)
            x_masked = x*mask_l
            relay_enc_l = self.relay_enc(x_masked, attn_item)
            relay_code[:, :, l+1, :] = relay_enc_l    # (B, n_patches, unit_trans_feat)
        
        relay_code = relay_code.view(B, self.n_patches, -1)
        relay_code = self.relay_pwnorm(relay_code)*np.sqrt(self.Pr/2)            # 2 here because of real numbers
        relay_code = torch.view_as_complex(relay_code.view(B, self.n_patches, -1, 2))

        return relay_code

    def forward(self, img, is_train = True):

        attn_item, g_sr, g_rd = self.gen_channel_coeff(is_train)

        B, C, H, W = img.shape

        # segment x -> (B, n_patches, n_feat)
        H_, W_ = int(H/self.n_part_dim), int(W/self.n_part_dim)
        x = rearrange(img, 'b c (p1 h) (p2 w) -> b (p1 p2) (h w c)', p1 = self.n_part_dim, p2 = self.n_part_dim, h = H_, w = W_)   # (B, n_patches, n_feat)

        if self.args.adapt:
            attn_item = attn_item.repeat(B, x.shape[1], 1)

        ### Source node
        x = self.enc(x, attn_item)                               # (B, n_patches, n_trans_feat = layers*unit_trans_feat)

        sig_s = x.view(B, self.n_patches, -1, 2)      # (B, n_patches, n_trans_feat/2, 2)
        sig_s = torch.view_as_complex(sig_s)
        sig_s = pwr_normalize(sig_s.view(B, -1))*np.sqrt(self.Ps)

        # to clarify the dimensions...
        sig_s = sig_s.view(B, self.n_patches, -1)

        noise_shape = sig_s.shape

        # S->R
        noise_sr = complex_sig(noise_shape, self.device)
        y_sr = torch.sqrt(g_sr)*sig_s + noise_sr
        
        # R->D
        y_sr = torch.view_as_real(y_sr).view(B, self.n_patches, self.layers*self.unit_trans_feat)
        relay_code = self.full_duplex_relaying(y_sr, attn_item)
        y_rd = torch.sqrt(g_rd)*relay_code

        # S->D
        noise_sd = complex_sig(noise_shape, self.device)
        y_sd = sig_s + noise_sd

        ### Receiver
        y_sd = y_sd + y_rd
        y_sd = torch.view_as_real(y_sd).view(B, self.n_patches, self.layers*self.unit_trans_feat)

        output = self.dec(y_sd, attn_item)

        # reshape output -> (B, 3, H, W)
        output = rearrange(output, 'b (p1 p2) (h w c) -> b c (p1 h) (p2 w)', p1 = self.n_part_dim, p2 = self.n_part_dim, h = H_, w = H_, c = C)

        return output


class RelayPFRD_Transformer_Dualinput_Adapt(Relay_Full_Duplex):
    def __init__(self,  args, enc, source_dec, relay_enc):
        super().__init__(args, enc, source_dec, relay_enc)

        self.layers = args.layers+1


        self.register_buffer('x_mean', torch.zeros(1))
        self.register_buffer('x_std', torch.zeros(1))

    def full_duplex_relaying(self, x, attn_item = None):
        # generate the 'parity' block-wise
        
        B, _, total_trans_feat = x.shape                           # （B, n_patches, layers*unit_trans_feat）
        relay_code = torch.zeros((B, self.n_patches, self.layers, self.unit_trans_feat)).to(x.device)

        # transformer mask
        for l in range(self.layers - 1):
            mask_l = self.gen_transformer_mask((B, self.n_patches, total_trans_feat), l+1)
            x_masked = x*mask_l                       
            relay_input = torch.cat((x_masked, relay_code.view(B, self.n_patches, -1)), dim = -1)
            relay_enc_l = self.relay_enc(relay_input, attn_item)
            relay_code[:, :, l+1, :] = relay_enc_l    # (B, n_patches, unit_trans_feat)
        
        relay_code = relay_code.view(B, self.n_patches, -1)
        relay_code = self.relay_pwnorm(relay_code)*np.sqrt(self.Pr/2)            # 2 here because of real numbers
        relay_code = torch.view_as_complex(relay_code.view(B, self.n_patches, -1, 2))

        return relay_code

    def forward(self, img, is_train = True):

        attn_item, g_sr, g_rd = self.gen_channel_coeff(is_train)

        B, C, H, W = img.shape

        # segment x -> (B, n_patches, n_feat)
        H_, W_ = int(H/self.n_part_dim), int(W/self.n_part_dim)
        x = rearrange(img, 'b c (p1 h) (p2 w) -> b (p1 p2) (h w c)', p1 = self.n_part_dim, p2 = self.n_part_dim, h = H_, w = W_)   # (B, n_patches, n_feat)

        if self.args.adapt:
            attn_item = attn_item.repeat(B, x.shape[1], 1)
        ### Source node
        x = self.enc(x, attn_item)                               # (B, n_patches, n_trans_feat = layers*unit_trans_feat)

        sig_s = x.view(B, self.n_patches, -1, 2)      # (B, n_patches, n_trans_feat/2, 2)
        sig_s = torch.view_as_complex(sig_s)
        sig_s = pwr_normalize(sig_s.view(B, -1))*np.sqrt(self.Ps)

        # to clarify the dimensions...
        sig_s = sig_s.view(B, self.n_patches, -1)

        noise_shape = sig_s.shape

        # S->R
        noise_sr = complex_sig(noise_shape, self.device)
        y_sr = torch.sqrt(g_sr)*sig_s + noise_sr
        
        # R->D
        y_sr = torch.view_as_real(y_sr).view(B, self.n_patches, self.layers*self.unit_trans_feat)
        relay_code = self.full_duplex_relaying(y_sr, attn_item)
        y_rd = torch.sqrt(g_rd)*relay_code

        # S->D
        noise_sd = complex_sig(noise_shape, self.device)
        y_sd = sig_s + noise_sd

        ### Receiver
        y_sd = y_sd + y_rd
        y_sd = torch.view_as_real(y_sd).view(B, self.n_patches, self.layers*self.unit_trans_feat)

        output = self.dec(y_sd, attn_item)

        # reshape output -> (B, 3, H, W)
        output = rearrange(output, 'b (p1 p2) (h w c) -> b c (p1 h) (p2 w)', p1 = self.n_part_dim, p2 = self.n_part_dim, h = H_, w = H_, c = C)

        return output


class RelayPFRD_Transformer_Dualinput_Full_Adapt(Relay_Full_Duplex):
    def __init__(self,  args, enc, source_dec, relay_enc):
        super().__init__(args, enc, source_dec, relay_enc)

        self.layers = args.layers+1


        self.register_buffer('x_mean', torch.zeros(1))
        self.register_buffer('x_std', torch.zeros(1))

    def gen_channel_coeff(self, is_train = False):

        if self.args.adapt and is_train:
            gamma1 = self.gamma1 + self.gamma_rng*(2*torch.rand(1)-1).to(self.device)
            gamma2 = self.gamma2 + self.gamma_rng*(2*torch.rand(1)-1).to(self.device)
            # randomly generate 
            p_rng1, p_rng2 = self.P_rng*(2*torch.rand(1)-1).to(self.device), self.P_rng*(2*torch.rand(1)-1).to(self.device)
            Ps = self.Ps * 10**(p_rng1/10)
            Pr = self.Pr * 10**(p_rng2/10)
        else:
            gamma1 = self.gamma1 + self.gamma_rng*torch.tensor([0]).to(self.device)
            gamma2 = self.gamma2 + self.gamma_rng*torch.tensor([0]).to(self.device)
            p_rng1, p_rng2 = self.P_rng*torch.tensor([0]).to(self.device), self.P_rng*torch.tensor([0]).to(self.device)
            Ps = self.Ps * 10**(p_rng1/10)
            Pr = self.Pr * 10**(p_rng2/10)
            
        attn_items = torch.cat((gamma1, gamma2, Ps, Pr)).unsqueeze(0).unsqueeze(0)                     # [1,1,4]
        g_sr = 10**(gamma1/10)
        g_rd = 10**(gamma2/10)


        return attn_items, g_sr, g_rd, Ps, Pr


    def full_duplex_relaying(self, x, attn_item = None):
        # generate the 'parity' block-wise
        
        B, _, total_trans_feat = x.shape                           # （B, n_patches, layers*unit_trans_feat）
        relay_code = torch.zeros((B, self.n_patches, self.layers, self.unit_trans_feat)).to(x.device)

        # transformer mask
        for l in range(self.layers - 1):
            mask_l = self.gen_transformer_mask((B, self.n_patches, total_trans_feat), l+1)
            x_masked = x*mask_l                       
            relay_input = torch.cat((x_masked, relay_code.view(B, self.n_patches, -1)), dim = -1)
            relay_enc_l = self.relay_enc(relay_input, attn_item)
            relay_code[:, :, l+1, :] = relay_enc_l    # (B, n_patches, unit_trans_feat)
        
        relay_code = relay_code.view(B, self.n_patches, -1)
        relay_code = self.relay_pwnorm(relay_code)           # 2 here because of real numbers
        relay_code = torch.view_as_complex(relay_code.view(B, self.n_patches, -1, 2))

        return relay_code

    def forward(self, img, is_train = True):

        attn_item, g_sr, g_rd, Ps, Pr = self.gen_channel_coeff(is_train)

        B, C, H, W = img.shape

        # segment x -> (B, n_patches, n_feat)
        H_, W_ = int(H/self.n_part_dim), int(W/self.n_part_dim)
        x = rearrange(img, 'b c (p1 h) (p2 w) -> b (p1 p2) (h w c)', p1 = self.n_part_dim, p2 = self.n_part_dim, h = H_, w = W_)   # (B, n_patches, n_feat)

        if self.args.adapt:
            attn_item = attn_item.repeat(B, x.shape[1], 1)
        ### Source node
        x = self.enc(x, attn_item)                               # (B, n_patches, n_trans_feat = layers*unit_trans_feat)

        sig_s = x.view(B, self.n_patches, -1, 2)      # (B, n_patches, n_trans_feat/2, 2)
        sig_s = torch.view_as_complex(sig_s)
        sig_s = pwr_normalize(sig_s.view(B, -1))*torch.sqrt(Ps)

        # to clarify the dimensions...
        sig_s = sig_s.view(B, self.n_patches, -1)

        noise_shape = sig_s.shape

        # S->R
        noise_sr = complex_sig(noise_shape, self.device)
        y_sr = torch.sqrt(g_sr)*sig_s + noise_sr
        
        # R->D
        y_sr = torch.view_as_real(y_sr).view(B, self.n_patches, self.layers*self.unit_trans_feat)
        relay_code = self.full_duplex_relaying(y_sr, attn_item)*torch.sqrt(Pr/2) 
        y_rd = torch.sqrt(g_rd)*relay_code

        # S->D
        noise_sd = complex_sig(noise_shape, self.device)
        y_sd = sig_s + noise_sd

        ### Receiver
        y_sd = y_sd + y_rd
        y_sd = torch.view_as_real(y_sd).view(B, self.n_patches, self.layers*self.unit_trans_feat)

        output = self.dec(y_sd, attn_item)

        # reshape output -> (B, 3, H, W)
        output = rearrange(output, 'b (p1 p2) (h w c) -> b c (p1 h) (p2 w)', p1 = self.n_part_dim, p2 = self.n_part_dim, h = H_, w = H_, c = C)

        return output




class RelayPFRD_Transformer_ResNet(Relay_Full_Duplex):
    def __init__(self,  args, enc, source_dec, relay_enc):
        super().__init__(args, enc, source_dec, relay_enc)

        self.layers = args.layers+1


        self.register_buffer('x_mean', torch.zeros(1))
        self.register_buffer('x_std', torch.zeros(1))

    def full_duplex_relaying(self, x, attn_item):
        # generate the 'parity' block-wise
        
        B, n_channel, _ = x.shape                           # （B, n_patches, layers*unit_trans_feat）
        relay_code = torch.zeros((B, n_channel, 8, 8)).to(x.device)
        x = x.view(B, n_channel, 8, 8)

        # transformer mask
        for l in range(self.layers - 1):
            mask_l = self.gen_resnet_mask((B, n_channel, 8, 8), l+1)
            x_masked = x*mask_l                       
            x_masked = x_masked.view(B, n_channel, 8, 8)
            relay_enc_l = self.relay_enc(x_masked, attn_item)
            relay_code[:, (l+1)*self.unit_trans_feat:(l+2)*self.unit_trans_feat, :, :] = relay_enc_l    # (B, n_patches, unit_trans_feat)
        
        relay_code = self.relay_pwnorm(relay_code)*np.sqrt(self.Pr/2)            # 2 here because of real numbers
        relay_code = torch.view_as_complex(relay_code.view(B, n_channel, -1, 2))

        return relay_code

    def forward(self, img, is_train = True):

        attn_item, g_sr, g_rd = self.gen_channel_coeff(is_train)
        attn_item = attn_item[0]

        B, C, H, W = img.shape

        # segment x -> (B, n_patches, n_feat)
        H_, W_ = int(H/self.n_part_dim), int(W/self.n_part_dim)
        x = img


        ### Source node
        x = self.enc(x, attn_item)                               # (B, n_channels, H_W_)
        n_channels = x.shape[1]

        sig_s = x.view(B, n_channels, -1, 2)          # (B, n_channels, H_W_/2, 2)
        sig_s = torch.view_as_complex(sig_s)
        sig_s = pwr_normalize(sig_s.view(B, -1))*np.sqrt(self.Ps)

        # to clarify the dimensions...
        sig_s = sig_s.view(B, n_channels, -1)

        noise_shape = sig_s.shape

        # S->R
        noise_sr = complex_sig(noise_shape, self.device)
        y_sr = torch.sqrt(g_sr)*sig_s + noise_sr
        
        # R->D
        y_sr = torch.view_as_real(y_sr).view(B, n_channels, -1)
        relay_code = self.full_duplex_relaying(y_sr, attn_item)
        y_rd = torch.sqrt(g_rd)*relay_code

        # S->D
        noise_sd = complex_sig(noise_shape, self.device)
        y_sd = sig_s + noise_sd

        ### Receiver
        y_sd = y_sd + y_rd
        y_sd = torch.view_as_real(y_sd).view(B, n_channels, -1).view(B, n_channels, 8, 8)

        output = self.dec(y_sd, attn_item)

        return output


class RelayAF_Vanilla(Relay_Full_Duplex):
    # Only need to implement full_duplex_relaying & forward function

    def __init__(self,  args, enc, dec, relay_enc):
        super().__init__(args, enc, dec, relay_enc)

        self.layers = args.layers+1

        self.register_buffer('x_mean', torch.zeros(1))
        self.register_buffer('x_std', torch.zeros(1))

    def full_duplex_relaying(self, x):

        relay_code = torch.zeros_like(x).to(x.device)

        #if self.args.zero_pad:
        relay_code[:,:,self.unit_trans_feat:] = x[:,:,0:-self.unit_trans_feat]
        '''
        else:
            # the parity encoder is used only for generating the first symbol
            mask = self.gen_mask(x.shape, 0)
            padding = self.relay_enc(mask)
            relay_code[:,0:self.unit_sym] = torch.view_as_complex(padding.view(B,-1,2))
            relay_code[:,self.unit_sym:] = x[:,0:-self.unit_sym]'''
        
        relay_code = self.relay_pwnorm_af(relay_code)*np.sqrt(self.Pr/2)
        relay_code = torch.view_as_complex(relay_code.view(x.shape[0], self.n_patches, -1, 2))

        return relay_code


    def forward(self, img, is_train = True):

        attn_item, g_sr, g_rd = self.gen_channel_coeff(is_train)

        B, C, H, W = img.shape

        # segment x -> (B, n_patches, n_feat)
        H_, W_ = int(H/self.n_part_dim), int(W/self.n_part_dim)
        x = rearrange(img, 'b c (p1 h) (p2 w) -> b (p1 p2) (h w c)', p1 = self.n_part_dim, p2 = self.n_part_dim, h = H_, w = W_)   # (B, n_patches, n_feat)

        if self.args.adapt:
            attn_item = attn_item.repeat(B, x.shape[1], 1)
        ### Source node
        x = self.enc(x, attn_item)                               # (B, n_patches, n_trans_feat = layers*unit_trans_feat)

        sig_s = x.view(B, self.n_patches, -1, 2)      # (B, n_patches, n_trans_feat/2, 2)
        sig_s = torch.view_as_complex(sig_s)
        sig_s = pwr_normalize(sig_s.view(B, -1))*np.sqrt(self.Ps)

        # to clarify the dimensions...
        sig_s = sig_s.view(B, self.n_patches, -1)

        noise_shape = sig_s.shape

        # S->R
        noise_sr = complex_sig(noise_shape, self.device)
        y_sr = torch.sqrt(g_sr)*sig_s + noise_sr
        
        # R->D
        y_sr = torch.view_as_real(y_sr).view(B, self.n_patches, self.layers*self.unit_trans_feat)
        relay_code = self.full_duplex_relaying(y_sr)
        y_rd = torch.sqrt(g_rd)*relay_code

        # S->D
        noise_sd = complex_sig(noise_shape, self.device)
        y_sd = sig_s + noise_sd

        ### Receiver
        y_sd = y_sd + y_rd
        y_sd = torch.view_as_real(y_sd).view(B, self.n_patches, self.layers*self.unit_trans_feat)

        output = self.dec(y_sd, attn_item)

        # reshape output -> (B, 3, H, W)
        output = rearrange(output, 'b (p1 p2) (h w c) -> b c (p1 h) (p2 w)', p1 = self.n_part_dim, p2 = self.n_part_dim, h = H_, w = H_, c = C)

        return output