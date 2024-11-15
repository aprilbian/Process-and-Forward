import numpy as np
from modules import *
from utils import complex_sig, pwr_normalize



class Relay_Half_Duplex(nn.Module):
    # A template for the following models; capable for CNN and transformers
    def __init__(self,  args, enc, dec, relay_enc):
        super().__init__()

        # CNN
        self.c_feat = args.cfeat
        self.c_out = args.cout

        # Transformer
        self.n_patches = args.n_patches                 # total # of patches
        self.n_feat = args.n_feat                       # # of features/patch
        self.n_part_dim = int(np.sqrt(self.n_patches))  # the partition is same for 2 dimensions
        self.hidden_size = args.hidden_size             # transformer dimension
        self.n_heads = args.n_heads                     # multihead attention
        self.n_layers = args.n_layers                   # # of transformer layers

        self.layers = args.layers
        self.layer_rng = args.layer_rng

        # varying features
        self.unit_trans_feat = args.unit_trans_feat           # bandwidth unit 
        self.max_trans_feat = args.max_trans_feat             # max # of transmit feats

        self.Ps = 10**(args.P/10)
        self.Pr = 10**(args.P/10)    # set them to be the same
        self.gamma1 = args.gamma1
        self.gamma2 = args.gamma2
        self.gamma_rng = args.gamma_rng


        self.unit_sym = int(self.unit_trans_feat*self.n_patches/2)
        self.args = args

        self.enc = enc                      # Source encoder
        self.relay_enc = relay_enc          # generate parity symbols
        self.dec = dec                      # Source decoder
        self.adapt = args.adapt
        self.device = args.device

    def gen_transformer_mask(self, layer):
        # mask for the transformer -- successively transmit patches
        mask_receive, mask_transmit = torch.ones(1, self.n_patches, self.max_trans_feat*self.unit_trans_feat).to(self.device), torch.ones(1, self.n_patches, self.max_trans_feat*self.unit_trans_feat).to(self.device)

        mask_receive[:,:,layer[0]*self.unit_trans_feat:] = 0
        mask_transmit[:,:,0:layer[0]*self.unit_trans_feat] = 0

        return mask_receive, mask_transmit
    
    def gen_channel_coeff(self, is_train = False):

        if self.args.adapt and is_train:
            gamma1 = self.gamma1 + self.gamma_rng*(2*torch.rand(1)-1).to(self.device)
            gamma2 = self.gamma2 + self.gamma_rng*(2*torch.rand(1)-1).to(self.device)
            layer = self.layers + torch.randint(-self.layer_rng, self.layer_rng + 1, (1,)).to(self.device)
        else:
            gamma1 = self.gamma1 + self.gamma_rng*torch.tensor([0]).to(self.device)
            gamma2 = self.gamma2 + self.gamma_rng*torch.tensor([0]).to(self.device)
            layer = self.layers + self.layer_rng*torch.tensor([0], dtype=int).to(self.device)

        attn_items = torch.cat((gamma1, gamma2, layer)).unsqueeze(0).unsqueeze(0)                     # [1,1,3]
        g_sr = 10**(gamma1/10)
        g_rd = 10**(gamma2/10)

        return attn_items, g_sr, g_rd, layer


table = [3, 3, 4, 4, 3, 3, 3, 4, 3, 3, 3, 4, 3, 3, 3, 3]
gamma_list = [0, 10/3, 20/3, 10]

class RelayHD_Transformer_Full_Adapt(Relay_Half_Duplex):
    def __init__(self,  args, enc, source_dec, relay_enc):
        super().__init__(args, enc, source_dec, relay_enc)


    def half_duplex_relaying(self, x, layer, attn_item = None):
        # x: [B, n_patches, self.layers*self.unit_trans_feat]
        
        B, n_patches, _ = x.shape                                  # （B, n_patches, layers*unit_trans_feat）

        receive_mask, transmit_mask = self.gen_transformer_mask(layer)

        rec_x = x*receive_mask
        
        relay_code = self.relay_enc(rec_x, attn_item)                             # （B, n_patches, (max_trans_feat - layers)*unit_trans_feat）

        trans_x = relay_code*transmit_mask
        
        relay_code = torch.view_as_complex(trans_x.view(B, -1, 2))
        relay_code = pwr_normalize(relay_code)*np.sqrt(self.Pr)

        # retrieve the dimension
        relay_code = torch.view_as_real(relay_code).view(B, n_patches, -1)

        return relay_code
    
    def half_duplex_relaying_AF(self, x, layer):
        # x: [B, n_patches, self.layers*self.unit_trans_feat]
        
        B, n_patches, _ = x.shape                                  # （B, n_patches, layers*unit_trans_feat）

        relay_code = torch.zeros_like(x).to(x.device)

        relay_code[:,:,layer*self.unit_trans_feat:2*layer*self.unit_trans_feat] = x[:,:,0:layer*self.unit_trans_feat]
        
        relay_code = torch.view_as_complex(relay_code.contiguous().view(B, -1, 2))
        relay_code = pwr_normalize(relay_code)*np.sqrt(self.Pr)

        # retrieve the dimension
        relay_code = torch.view_as_real(relay_code).view(B, n_patches, -1)

        return relay_code

    def determine_layer(self, gamma1, gamma2):
        # x: [B, n_patches, self.layers*self.unit_trans_feat]
        
        # determine the index
        index = int(gamma1/(10/3))*4 + int(gamma2/(10/3))
        layer = table[index]

        return layer

    
    def gen_channel_coeff_v1(self, is_train = False):

        if self.args.adapt and is_train:
            index1, index2 = torch.randint(0, 4, (1,)), torch.randint(0, 4, (1,))
            gamma1 = gamma_list[index1] + self.gamma_rng*torch.tensor([0]).to(self.device)
            gamma2 = gamma_list[index2] + self.gamma_rng*torch.tensor([0]).to(self.device)
        else:
            gamma1 = self.gamma1 + self.gamma_rng*torch.tensor([0]).to(self.device)
            gamma2 = self.gamma2 + self.gamma_rng*torch.tensor([0]).to(self.device)

        layer = torch.tensor([self.determine_layer(gamma1, gamma2)]).to(self.device)

        attn_items = torch.cat((gamma1, gamma2, layer)).unsqueeze(0).unsqueeze(0)                     # [1,1,3]
        g_sr = 10**(gamma1/10)
        g_rd = 10**(gamma2/10)

        return attn_items, g_sr, g_rd, layer

    def gen_channel_coeff(self, is_train = False):

        if self.args.adapt and is_train:
            gamma1 = self.gamma1 + (self.gamma_rng-0.01)*(2*torch.rand(1)-1).to(self.device)
            gamma2 = self.gamma2 + (self.gamma_rng-0.01)*(2*torch.rand(1)-1).to(self.device)
        else:
            gamma1 = self.gamma1 + self.gamma_rng*torch.tensor([0]).to(self.device)
            gamma2 = self.gamma2 + self.gamma_rng*torch.tensor([0]).to(self.device)

        layer = torch.tensor([self.determine_layer(gamma1, gamma2)]).to(self.device)

        attn_items = torch.cat((gamma1, gamma2, layer)).unsqueeze(0).unsqueeze(0)                     # [1,1,3]
        g_sr = 10**(gamma1/10)
        g_rd = 10**(gamma2/10)

        return attn_items, g_sr, g_rd, layer

    def forward(self, img, is_train = True):

        attn_item, g_sr, g_rd, layer = self.gen_channel_coeff(is_train)
        #attn_item, g_sr, g_rd, layer = self.gen_channel_coeff_v1(is_train)

        B, C, H, W = img.shape

        # segment x -> (B, n_patches, n_feat)
        H_, W_ = int(H/self.n_part_dim), int(W/self.n_part_dim)
        x = rearrange(img, 'b c (p1 h) (p2 w) -> b (p1 p2) (h w c)', p1 = self.n_part_dim, p2 = self.n_part_dim, h = H_, w = W_)   # (B, n_patches, n_feat)

        if self.args.adapt:
            attn_item = attn_item.repeat(B, x.shape[1], 1)
        ### Source node
            x = self.enc(x, attn_item)                               # (B, n_patches, n_trans_feat)
        else:
            x = self.enc(x)

        sig_s = x.view(B, self.n_patches, -1, 2)      # (B, n_patches, n_trans_feat/2, 2)
        sig_s = torch.view_as_complex(sig_s)
        sig_s = pwr_normalize(sig_s.view(B, -1))*np.sqrt(self.Ps)

        sig_s = sig_s.view(B, self.n_patches, -1)

        noise_shape = sig_s.shape

        # S->R
        noise_sr = complex_sig(noise_shape, self.device)
        y_sr = torch.sqrt(g_sr)*sig_s + noise_sr
        
        # R->D
        y_sr = torch.view_as_real(y_sr).view(B, self.n_patches, -1)
        if self.args.relay_mode == 'AF':
            #assert 2*self.layers == self.max_trans_feat, 'relay-receive period should equal to the relay-transmit period'
            relay_code = self.half_duplex_relaying_AF(y_sr, 3)
        else:
            relay_code = self.half_duplex_relaying(y_sr, layer, attn_item)
        y_rd = torch.sqrt(g_rd)*relay_code

        # S->D
        noise_sd = complex_sig(noise_shape, self.device)
        y_sd = sig_s + noise_sd

        ### Receiver
        y_sd = torch.view_as_real(y_sd).view(B, self.n_patches, -1)
        y_sd = y_sd + y_rd

        output = self.dec(y_sd, attn_item)

        # reshape output -> (B, 3, H, W)
        output = rearrange(output, 'b (p1 p2) (h w c) -> b c (p1 h) (p2 w)', p1 = self.n_part_dim, p2 = self.n_part_dim, h = H_, w = H_, c = C)

        return output



class RelayHD_Transformer_fading(Relay_Half_Duplex):
    def __init__(self,  args, enc, source_dec, relay_enc):
        super().__init__(args, enc, source_dec, relay_enc)

    
    def gen_channel_coeff(self):

        # here, we generate the Rayleigh fading amplitudes

        h_sr, h_rd, h_sd = torch.randn(1, dtype=torch.cfloat).to(self.device), torch.randn(1, dtype=torch.cfloat).to(self.device), torch.randn(1, dtype=torch.cfloat).to(self.device)
        h_sr, h_rd, h_sd = torch.abs(h_sr), torch.abs(h_rd), torch.abs(h_sd)

        gamma1 = self.gamma1 + 20*torch.log10(h_sr)
        gamma2 = self.gamma2 + 20*torch.log10(h_rd)
        gamma3 = 20*torch.log10(h_sd)
        
        attn_items = torch.cat((gamma1, gamma2, gamma3)).unsqueeze(0).unsqueeze(0)                     # [1,1,4]
        g_sr = 10**(gamma1/10)
        g_rd = 10**(gamma2/10)
        g_sd = 10**(gamma3/10)


        return attn_items, g_sr, g_rd, g_sd


    def half_duplex_relaying(self, x, layer, attn_item = None):
        # x: [B, n_patches, self.layers*self.unit_trans_feat]
        
        B, n_patches, _ = x.shape                                  # （B, n_patches, layers*unit_trans_feat）

        receive_mask, transmit_mask = self.gen_transformer_mask(layer)

        rec_x = x*receive_mask
        
        relay_code = self.relay_enc(rec_x, attn_item)                             # （B, n_patches, (max_trans_feat - layers)*unit_trans_feat）

        trans_x = relay_code*transmit_mask
        
        relay_code = torch.view_as_complex(trans_x.view(B, -1, 2))
        relay_code = pwr_normalize(relay_code)*np.sqrt(self.Pr)

        # retrieve the dimension
        relay_code = torch.view_as_real(relay_code).view(B, n_patches, -1)

        return relay_code
    
    def half_duplex_relaying_AF(self, x, layer):
        # x: [B, n_patches, self.layers*self.unit_trans_feat]
        
        B, n_patches, _ = x.shape                                  # （B, n_patches, layers*unit_trans_feat）

        relay_code = torch.zeros_like(x).to(x.device)

        relay_code[:,:,layer*self.unit_trans_feat:2*layer*self.unit_trans_feat] = x[:,:,0:layer*self.unit_trans_feat]
        
        relay_code = torch.view_as_complex(relay_code.contiguous().view(B, -1, 2))
        relay_code = pwr_normalize(relay_code)*np.sqrt(self.Pr)

        # retrieve the dimension
        relay_code = torch.view_as_real(relay_code).view(B, n_patches, -1)

        return relay_code

    def forward(self, img, is_train = True):

        attn_item, g_sr, g_rd, g_sd = self.gen_channel_coeff(is_train)

        B, C, H, W = img.shape

        # segment x -> (B, n_patches, n_feat)
        H_, W_ = int(H/self.n_part_dim), int(W/self.n_part_dim)
        x = rearrange(img, 'b c (p1 h) (p2 w) -> b (p1 p2) (h w c)', p1 = self.n_part_dim, p2 = self.n_part_dim, h = H_, w = W_)   # (B, n_patches, n_feat)

        if self.args.adapt:
            attn_item = attn_item.repeat(B, x.shape[1], 1)
        ### Source node
            x = self.enc(x, attn_item)                               # (B, n_patches, n_trans_feat)
        else:
            x = self.enc(x)

        sig_s = x.view(B, self.n_patches, -1, 2)      # (B, n_patches, n_trans_feat/2, 2)
        sig_s = torch.view_as_complex(sig_s)
        sig_s = pwr_normalize(sig_s.view(B, -1))*np.sqrt(self.Ps)

        sig_s = sig_s.view(B, self.n_patches, -1)

        noise_shape = sig_s.shape

        # S->R
        noise_sr = complex_sig(noise_shape, self.device)
        y_sr = torch.sqrt(g_sr)*sig_s + noise_sr
        
        # R->D
        y_sr = y_sr/torch.sqrt(g_sr)
        y_sr = torch.view_as_real(y_sr).view(B, self.n_patches, -1)
        if self.args.relay_mode == 'AF':
            #assert 2*self.layers == self.max_trans_feat, 'relay-receive period should equal to the relay-transmit period'
            relay_code = self.half_duplex_relaying_AF(y_sr, 3)
        else:
            relay_code = self.half_duplex_relaying(y_sr, 3, attn_item)
        y_rd = torch.sqrt(g_rd)*relay_code

        # S->D
        noise_sd = complex_sig(noise_shape, self.device)
        y_sd = torch.sqrt(g_sd)*sig_s + noise_sd

        ### Receiver
        y_sd = torch.view_as_real(y_sd).view(B, self.n_patches, -1)
        y_sd = y_sd + y_rd

        output = self.dec(y_sd, attn_item)

        # reshape output -> (B, 3, H, W)
        output = rearrange(output, 'b (p1 p2) (h w c) -> b c (p1 h) (p2 w)', p1 = self.n_part_dim, p2 = self.n_part_dim, h = H_, w = H_, c = C)

        return output