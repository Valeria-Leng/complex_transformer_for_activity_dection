import torch
import numpy as np
from torch import nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            y_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.y_dim = y_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
        
        self.Wy_query =  nn.Parameter(torch.Tensor(n_heads, y_dim, key_dim))
        self.Wy_key = nn.Parameter(torch.Tensor(n_heads, y_dim, key_dim))
        self.Wy_val = nn.Parameter(torch.Tensor(n_heads, y_dim, val_dim))
 


        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, input_dim))
        self.Wy_out = nn.Parameter(torch.Tensor(n_heads, val_dim, y_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, y, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        #if h is None:
        h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        y_dim = y.size(2)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"
        assert y_dim == self.y_dim, "Wrong embedding dimension of input y"
        

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)
        yflat = y.contiguous().view(-1, y_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_y = (self.n_heads, batch_size, 1, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, batch_size, n_query, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        Qy = torch.matmul(yflat, self.Wy_query).view(shp_y)
        Q = torch.cat((Q,Qy), dim=2)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        Ky = torch.matmul(yflat, self.Wy_key).view(shp_y)
        K = torch.cat((K,Ky), dim=2)
        V = torch.matmul(hflat, self.W_val).view(shp)
        Vy = torch.matmul(yflat, self.Wy_val).view(shp_y)
        V = torch.cat((V,Vy), dim=2)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        #compatibility_y = self.norm_factor * torch.matmul(Qy, K.transpose(2, 3))
        
        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, 1, n_query+1, graph_size+1).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)
        #attn_y = torch.softmax(compatibility_y, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads_all = torch.matmul(attn, V)
        heads = heads_all[:,:,:-1,:]
        heads_y = heads_all[:,:,-1:,:]
       # heads_y = torch.matmul(attn_y, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.input_dim)
        ).view(batch_size, n_query, self.input_dim)
        
        out_y = torch.mm(
            heads_y.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.Wy_out.view(-1, self.y_dim)
        ).view(batch_size, 1, self.y_dim)

        # Alternative:
        # headst = heads.transpose(0, 1)  # swap the dimensions for batch and heads to align it for the matmul
        # # proj_h = torch.einsum('bhni,hij->bhnj', headst, self.W_out)
        # projected_heads = torch.matmul(headst, self.W_out)
        # out = torch.sum(projected_heads, dim=1)  # sum across heads

        # Or:
        # out = torch.einsum('hbni,hij->bnj', heads, self.W_out)

        return out, out_y


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class USREncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            y_dim,
            x_dim,
            feed_forward_hidden,
            node_dim=None,
            normalization='batch'
    ):
        super(USREncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, x_dim) if node_dim is not None else None
       # self.nlx = Normalization(embed_dim, normalization) if node_dim is not None else None
        self.mha =  MultiHeadAttention(
                    n_heads,
                    input_dim=x_dim,
                    y_dim=y_dim,
                    embed_dim=embed_dim
                )
        self.nl1 = Normalization(x_dim, normalization)
        self.nly = Normalization(y_dim, normalization)
        self.ff = nn.Sequential(
                    nn.Linear(x_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, x_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(x_dim, x_dim)
        self.ffy = nn.Sequential(
                    nn.Linear(y_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, y_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(y_dim, y_dim)        
        self.nl2 = Normalization(x_dim, normalization)
        self.nly2 = Normalization(y_dim, normalization)

    def forward(self, x, y, mask=None):

        #assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x
        h_new, y_new = self.mha(h,y,mask)
        h = h+h_new
        y = y+y_new
        h = self.nl1(h)
        y = self.nly(y)
        #h = self.ff(torch.cat((h,y), 2)) + h
        h = self.ff(h) + h
        y = self.ffy(y) + y
        h = self.nl2(h)
        y = self.nly2(y)

        return (
            h,  # (batch_size, graph_size, embed_dim)
           # h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
            y # (batch_size, 1, y_dim)
        )
