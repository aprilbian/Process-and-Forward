import ipdb
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor

class AFModule(nn.Module):
    def __init__(self, c_in):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=c_in+3,
                      out_features=c_in),

            nn.LeakyReLU(),

            nn.Linear(in_features=c_in,
                      out_features=c_in),

            nn.Sigmoid()
        )

    def forward(self, x, snr):
        B, _, H, W = x.size()
        context = torch.mean(x, dim=(2, 3))
        snr_context = snr.repeat_interleave(B // snr.size(0), dim=0)

        # snr_context = torch.ones(B, 1, requires_grad=True).to(x.device) * snr
        context_input = torch.cat((context, snr_context), dim=1)
        atten_weights = self.layers(context_input).view(B, -1, 1, 1)
        atten_mask = torch.repeat_interleave(atten_weights, H, dim=2)
        atten_mask = torch.repeat_interleave(atten_mask, W, dim=3)
        out = atten_mask * x
        return out




class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


########### The transformer blocks implemented by myself ##############



from einops import rearrange

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class position_embedding(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.max_position_embed = args.n_patches
        self.hidden_size = args.hidden_size

        self.position_encoder = nn.Embedding(self.max_position_embed, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size, 1e-12)
        self.drop_out = nn.Dropout(args.dropout_prob)
    
    def forward(self, x):

        B, n_patches, _ = x.shape     # (B, n_patches, hidden_size)
        indexes = torch.arange(n_patches).unsqueeze(0).expand((B, n_patches))   # (B, n_patches)
        indexes = indexes.to(x.device)

        pos_info = self.position_encoder(indexes)
        x = x + pos_info
        x = self.drop_out(self.layer_norm(x))

        return x


class transformer_encoderlayer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.n_heads = args.n_heads
        self.hidden_size = args.hidden_size
        self.attn_size = int(self.hidden_size/self.n_heads)
        self.ff_size = args.feedforward_size

        assert self.hidden_size%self.n_heads == 0

        # multihead_attention layer
        self.qnet = nn.Linear(self.hidden_size, self.hidden_size)
        self.knet = nn.Linear(self.hidden_size, self.hidden_size)
        self.vnet = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(args.dropout_prob)

        self.mha_dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.mha_layernorm = nn.LayerNorm(self.hidden_size, 1e-12)

        # feed forward layer
        self.ff_dense1 = nn.Linear(self.hidden_size, self.ff_size)
        self.ff_dense2 = nn.Linear(self.ff_size, self.hidden_size)
        self.ff_layernorm = nn.LayerNorm(self.hidden_size, 1e-12)
    
    def multihead_attention(self, x, mask = None):

        B, n_patches, _ = x.shape     # (B, n_patches, hidden_size)

        query, key, value = self.qnet(x), self.knet(x), self.vnet(x)    # (B, n_patches, hidden_sz)

        # multihead  ->  (B, n_heads, n_patches, attn_size)
        query = query.view(B, n_patches, self.n_heads, self.attn_size).permute(0, 2, 1, 3) 
        key = key.view(B, n_patches, self.n_heads, self.attn_size).permute(0, 2, 3, 1)       # different from the others
        value = value.view(B, n_patches, self.n_heads, self.attn_size).permute(0, 2, 1, 3) 

        # inner-product & softmax & mixing
        inner_product = torch.matmul(query, key)                # (B, n_heads, n_patches, n_patches)
        inner_product = inner_product/np.sqrt(self.attn_size)   # normalize

        if mask != None:
            inner_product = inner_product + mask
        
        prob = nn.Softmax(dim = -1)(inner_product)          # each row sums up = 1 
        prob = self.dropout(prob)

        weighted_value = torch.matmul(prob, value)               # (B, n_heads, n_patches, hidden_sz)
        weighted_value = weighted_value.permute(0, 2, 1, 3).contiguous().view(B, n_patches, -1)

        weighted_value = self.mha_layernorm(self.dropout(self.mha_dense(weighted_value)) + x)
        return weighted_value
    
    def feedforward_layer(self, x):

        hidden_state = gelu(self.ff_dense1(x))
        hidden_state = self.dropout(self.ff_dense2(hidden_state))
        hidden_state = self.ff_layernorm(hidden_state + x)

        return hidden_state

    
    def forward(self, x, mask = None):

        # multihead attention layer
        x = self.multihead_attention(x, mask)

        # feedforward layer
        x = self.feedforward_layer(x)

        return x
        
        
class transformer_encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.n_layers = args.n_layers

        self.neural_layers = nn.ModuleList([transformer_encoderlayer(args) for _ in range(self.n_layers)])
    
    def forward(self, x):

        for _, layer_module in enumerate(self.neural_layers):
            x = layer_module(x)
        
        return x



class tf_encoder(nn.Module):
    def __init__(self,  args):
        super().__init__()

        self.n_patches = args.n_patches              # total # of patches
        self.n_feat = args.n_feat                    # # of features/patch
        self.hidden_size = args.hidden_size          # transformer dimension
        self.n_heads = args.n_heads                  # multihead attention
        self.n_layers = args.n_layers                # # of transformer layers
        self.n_trans_feat = args.n_trans_feat        # # of transmit features/patch
        self.max_trans_feat = args.max_trans_feat    # # of total bandwidth
        self.unit_trans_feat = args.unit_trans_feat  # bandwidth unit 

        self.n_part_dim = int(np.sqrt(self.n_patches))  # the partition is same for 2 dimensions

        self.adapt = args.adapt                    # if adaptive to bandwidth & SNR
        self.n_adapt_embed = args.n_adapt_embed    # embedding size for bandwidth & SNR

        self.args = args

        ## encoder consists -- 1. linear projection; 2. position encoding
        ## 3. Transformer layers 4. final layer

        # linear projection
        if self.adapt:
            self.linear_proj = nn.Linear(self.n_feat + self.n_adapt_embed, self.hidden_size)
        else:
            self.linear_proj = nn.Linear(self.n_feat, self.hidden_size)
        self.layer_norm1  = nn.LayerNorm(self.hidden_size, 1e-12)

        # position encoding
        self.pos_embeding = position_embedding(args)

        # transformer
        self.transformer_encoder = transformer_encoder(args)

        # final dense layers
        #if self.adapt:
        #    self.final_layer = nn.Linear(self.hidden_size, self.max_trans_feat*self.unit_trans_feat)
        #else:
        self.final_layer = nn.Linear(self.hidden_size, self.n_trans_feat)
    
    def forward(self, x, adapt_embedding = None):
        # adapt_embedding is designed for adaptive bandwidth & snr

        # linear proj
        if self.adapt:
            x = torch.cat((x, adapt_embedding), dim = 2)
        x = self.layer_norm1(gelu(self.linear_proj(x)))    # (B, n_patches, hidden_sz)

        # position embedding
        x = self.pos_embeding(x)

        # transformer
        x = self.transformer_encoder(x)

        # final dense layer
        x = self.final_layer(x)

        return x

class tf_decoder(nn.Module):
    def __init__(self,  args):
        super().__init__()

        self.n_patches = args.n_patches              # total # of patches
        self.n_feat = args.n_feat                    # # of features/patch
        self.hidden_size = args.hidden_size          # transformer dimension
        self.max_trans_feat = args.max_trans_feat    # # of total bandwidth
        self.unit_trans_feat = args.unit_trans_feat  # bandwidth unit 

        self.n_part_dim = int(np.sqrt(self.n_patches))  # the partition is same for 2 dimensions

        self.args = args

        self.adapt = args.adapt
        self.n_adapt_embed = args.n_adapt_embed

        ## decoder consists -- 1. linear projection; 2. position encoding
        ## 3. Transformer layers 4. final layer

        # linear projection
        if self.adapt:
            self.linear_proj = nn.Linear(self.max_trans_feat*self.unit_trans_feat + self.n_adapt_embed, self.hidden_size)
        else:
            self.linear_proj = nn.Linear(self.max_trans_feat*self.unit_trans_feat, self.hidden_size)
        self.layer_norm1  = nn.LayerNorm(self.hidden_size, 1e-12)

        # position encoding
        self.pos_embeding = position_embedding(args)

        # transformer
        self.transformer_encoder = transformer_encoder(args)

        # final dense layers
        self.final_layer = nn.Linear(self.hidden_size, self.n_feat)
    
    def forward(self, x, adapt_embedding = None):
        
        # linear proj
        if self.adapt:
            x = torch.cat((x, adapt_embedding), dim = 2)
        x = self.layer_norm1(gelu(self.linear_proj(x)))    # (B, n_patches, hidden_sz)

        # position embedding
        x = self.pos_embeding(x)

        # transformer
        x = self.transformer_encoder(x)

        # final dense layer
        x = self.final_layer(x)

        return x











class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.percentage = percentage
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.best_epoch = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics, epoch):
        if self.patience == 0:
            return False, self.best, self.best_epoch, self.num_bad_epochs

        if self.best is None:
            self.best = metrics
            self.best_epoch = epoch
            return False, self.best, self.best_epoch, 0

        if torch.isnan(metrics):
            return True, self.best, self.best_epoch, self.num_bad_epochs

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
            self.best_epoch = epoch
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True, self.best, self.best_epoch, self.num_bad_epochs

        return False, self.best, self.best_epoch, self.num_bad_epochs

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

    def get_state_dict(self):
        state_dict = {
            'best': self.best,
            'best_epoch': self.best_epoch,
            'num_bad_epochs': self.num_bad_epochs,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.best = state_dict['best']
        self.best_epoch = state_dict['best_epoch']
        self.num_bad_epochs = state_dict['num_bad_epochs']

    def reset(self):
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self.best_epoch = None
        self._init_is_better(self.mode, self.min_delta, self.percentage)