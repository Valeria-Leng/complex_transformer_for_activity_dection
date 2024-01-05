import torch
from torch import nn
from complexPyTorch.complexLayers import ComplexConv2d, ComplexLinear
from complexPyTorch.complexLayers import ComplexDropout, NaiveComplexBatchNorm2d
from complexPyTorch.complexLayers import ComplexBatchNorm1d
from complexPyTorch.complexFunctions import complex_relu
from torch.nn.functional import softmax, relu, sigmoid

class ComplexConv1d(nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self,x):
        input_r = x.real
        input_i = x.imag
        out_r = self.conv_r(input_r)-self.conv_i(input_i)
        out_i = self.conv_r(input_i)+self.conv_i(input_r)

        return out_r + 1j * out_i


# class ComplexLinear(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(ComplexLinear, self).__init__()
#         self.fc_r = nn.Linear(in_features, out_features)
#         self.fc_i = nn.Linear(in_features, out_features)

#     def forward(self, input):
#         dim = input.shape[0]
#         n_input = torch.cat((input.real, input.imag))
#         out_r = self.fc_r(n_input) #[rW_r   i_W_r]
#         out_i = self.fc_i(n_input) #[rW_i   i_W_i]
#         return (out_r[:dim]-out_i[dim:]) + 1j*(out_r[dim:]+out_i[:dim])

class Complex_Transformer(nn.Module):
    def __init__(self,  y_dim = 64, x_dim= 64, embed_dim=64, out_dim=100, node_dim=12, num_heads=4, device='cuda'):
        super().__init__()
        # # self.dlconv = ComplexConv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        # self.dlconv1 = ComplexConv1d(in_channels=1, out_channels=2, kernel_size=6, stride=2, padding=2)
        # self.dlconv2 = ComplexConv1d(in_channels=2, out_channels=4, kernel_size=6, stride=2, padding=2)
        self.cov= ComplexLinear(x_dim, embed_dim)
        self.usr1 = Transformer_Encoder(embed_dim=embed_dim, node_dim=node_dim, num_heads=num_heads, device='cuda')
        self.usr2 = Transformer_Encoder(embed_dim = embed_dim, A_dim=x_dim, num_heads=num_heads, device='cuda')
        self.out = Out(num_heads =num_heads, embed_dim = embed_dim, y_dim = y_dim, x_dim= x_dim, device='cuda')
 
        self.outproj = ComplexLinear(embed_dim, out_dim)
        # self.finalconv = ComplexConv1d(in_channels=4, out_channels=1, kernel_size=3, padding=1)
        self.C2R = nn.Linear(2,1)
        # self.out_dropout = ComplexDropout(res_dropout)
        self.device = device
    def forward(self, A, x):
        #A: B,C,L = 256, 20, 8 
        #x: B,L = 256, 64
        #output: 256, 20
        # x = self.cov(x).unsqueeze(1) #256, 64 -> 256, 1, 64 
        A, x = self.usr1(A, x)     #256, 20, 64  #256, 1, 64  
        A, x = self.usr2(A, x)    #256, 20, 64  #256, 1, 64
        out = self.out(A, x)  # B,20
        # out = torch.view_as_real(out)
        # out = self.C2R(out).squeeze(dim=-1) #B,20, 2->1
        return out#256, 20
     

class Complex_LayerNorm(nn.Module):

    def __init__(self, embed_dim=None, eps=1e-05, elementwise_affine=True, device='cuda'):
        super().__init__()
        assert not(elementwise_affine and embed_dim is None), 'Give dimensions of learnable parameters or disable them'
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.embed_dim = embed_dim
            self.register_parameter(name='weights', param=torch.nn.Parameter(torch.empty([2, 2], dtype=torch.complex64)))
            self.register_parameter(name='bias', param=torch.nn.Parameter(torch.zeros(embed_dim, dtype=torch.complex64)))
            self.weights = torch.nn.Parameter(torch.eye(2))
            self.weights = torch.nn.Parameter((torch.Tensor([1, 1, 0]).repeat([embed_dim, 1])).unsqueeze(-1))
            self.bias = torch.nn.Parameter(torch.zeros([1, 1, embed_dim], dtype=torch.complex64))
        self.eps = eps

    def forward(self, input):

        ev = torch.unsqueeze(torch.mean(input, dim=-1), dim=-1)
        var_real = torch.unsqueeze(torch.unsqueeze(torch.var(input.real, dim=-1), dim=-1), dim=-1)
        var_imag = torch.unsqueeze(torch.unsqueeze(torch.var(input.imag, dim=-1), dim=-1), dim=-1)

        input = input - ev
        cov = torch.unsqueeze(torch.unsqueeze(torch.mean(input.real * input.imag, dim=-1), dim=-1), dim=-1)
        cov_m_0 = torch.cat((var_real, cov), dim=-1)
        cov_m_1 = torch.cat((cov, var_imag), dim=-1)
        cov_m = torch.unsqueeze(torch.cat((cov_m_0, cov_m_1), dim=-2), dim=-3)
        in_concat = torch.unsqueeze(torch.cat((torch.unsqueeze(input.real, dim=-1), torch.unsqueeze(input.imag, dim=-1)), dim=-1), dim=-1)

        cov_sqr = self.sqrt_2x2(cov_m).cuda()

        # out = self.inv_2x2(cov_sqr).matmul(in_concat)  # [..., 0]
        if self.elementwise_affine:
            real_var_weight = (self.weights[:, 0, :] ** 2).unsqueeze(-1).unsqueeze(0)
            imag_var_weight = (self.weights[:, 1, :] ** 2).unsqueeze(-1).unsqueeze(0)
            cov_weight = (torch.sigmoid(self.weights[:, 2, :].unsqueeze(-1).unsqueeze(0)) - 0.5) * 2 * torch.sqrt(real_var_weight * imag_var_weight)
            weights_mult = torch.cat([torch.cat([real_var_weight, cov_weight], dim=-1), torch.cat([cov_weight, imag_var_weight], dim=-1)], dim=-2).unsqueeze(0).cuda()
            mult_mat = self.sqrt_2x2(weights_mult).matmul(self.inv_2x2(cov_sqr))
            out = mult_mat.matmul(in_concat)  # makes new cov_m = self.weights
        else:
            out = self.inv_2x2(cov_sqr).matmul(in_concat)  # [..., 0]
        out = out[..., 0, 0] + 1j * out[..., 1, 0]  # torch.complex(out[..., 0], out[..., 1]) not used because of memory requirements
        if self.elementwise_affine:
            return out + self.bias.cuda()
        return out

    def inv_2x2(self, input):
        a = torch.unsqueeze(torch.unsqueeze(input[..., 0, 0], dim=-1), dim=-1)
        b = torch.unsqueeze(torch.unsqueeze(input[..., 0, 1], dim=-1), dim=-1)
        c = torch.unsqueeze(torch.unsqueeze(input[..., 1, 0], dim=-1), dim=-1)
        d = torch.unsqueeze(torch.unsqueeze(input[..., 1, 1], dim=-1), dim=-1)
        divisor = a * d - b * c
        mat_1 = torch.cat((d, -b), dim=-2)
        mat_2 = torch.cat((-c, a), dim=-2)
        mat = torch.cat((mat_1, mat_2), dim=-1)
        return mat / divisor

    def sqrt_2x2(self, input):
        a = torch.unsqueeze(torch.unsqueeze(input[..., 0, 0], dim=-1), dim=-1)
        b = torch.unsqueeze(torch.unsqueeze(input[..., 0, 1], dim=-1), dim=-1)
        c = torch.unsqueeze(torch.unsqueeze(input[..., 1, 0], dim=-1), dim=-1)
        d = torch.unsqueeze(torch.unsqueeze(input[..., 1, 1], dim=-1), dim=-1)

        s = torch.sqrt(a * d - b * c)  # sqrt(det)
        t = torch.sqrt(a + d + 2 * s)  # sqrt(trace + 2 * sqrt(det))
        # maybe use 1/t * (M + sI) later, see Wikipedia

        return torch.cat((torch.cat((a + s, b), dim=-2), torch.cat((c, d + s), dim=-2)), dim=-1) / t

class FeeaforwardNN(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.fc1 = ComplexLinear(embed_dim, embed_dim*4)
        self.fc2 = ComplexLinear(embed_dim*4, embed_dim)
        # self.dropout = ComplexDropout(relu_dropout)
    def forward(self, src):
        #####################
        # fc1 --> relu --> fc1
        src = self.fc1(src) #128, 4, 256
        src = complex_relu(src)
        # src = self.dropout(src)
        src = self.fc2(src) #128, 4, 64
        return src
        #####################        

class attention(nn.Module):
    def __init__(self, input_dim=64, y_dim=64, embed_dim=64, num_heads=4, val_dim=None, key_dim=None, Glimpse=None) :
        super().__init__()
        if val_dim is None:
            val_dim = embed_dim // num_heads
        if key_dim is None:
            key_dim = val_dim
        self.y_dim = y_dim
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.Glimpse = Glimpse
        assert self.val_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.val_dim ** -0.5 #4
        self.pro_q = ComplexLinear(self.embed_dim, self.embed_dim)
        self.pro_k = ComplexLinear(self.embed_dim, self.embed_dim)
        self.pro_v = ComplexLinear(self.embed_dim, self.embed_dim)
        self.out_proj = ComplexLinear(embed_dim, embed_dim)

    def forward(self, q, y): #256, 100, 64 #256, 1, 64
        h = q
        B, n_query, input_dim = q.size() #256, 100, 64
        # n_query = q.size(1) #100
        y_dim = y.size(2) #64
        # assert q.size(0) == B
        # assert q.size(2) == input_dim #64
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"
        assert y_dim == self.y_dim, "Wrong embedding dimension of input y"
        ###########################################
        #multi_head_attention
        #split Q,K,V  (n_heads, batch_size, n_query, key/val_size) #4, 256, 21, 16
        Q = self.pro_q(q)
        Qy = self.pro_q(y)
        if self.Glimpse is None:
            Q = torch.cat((Q,Qy), dim=-2).view(B, n_query+1, self.num_heads, -1).contiguous().permute(2, 0, 1, 3) 
        else:
            Q = Qy.view(B, 1, self.num_heads, -1).contiguous().permute(2, 0, 1, 3) 
        K = self.pro_k(h)
        Ky = self.pro_k(y)
        K = torch.cat((K,Ky), dim=-2).view(B, n_query+1, self.num_heads, -1).contiguous().permute(2, 0, 1, 3)
        V = self.pro_v(h)
        Vy = self.pro_v(y)
        V = torch.cat((V,Vy), dim=-2).view(B, n_query+1, self.num_heads, -1).contiguous().permute(2, 0, 1, 3)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        attn_weights = self.scaling * torch.matmul(Q, torch.conj_physical(K).transpose(-1, -2)) #4, 256, 101, 101
        

        #softmax
        attn_weights = self.softmax_real(attn_weights) #4, 256, 101, 101 dim=-1
        attn = torch.matmul(attn_weights, V) #4, 256, 101, 16 
        heads_all = attn.permute(1, 2, 0, 3).contiguous().view(B, -1, self.num_heads*self.key_dim)
        heads_all = self.out_proj(heads_all) #256, 101, 64 
        if heads_all.size(1) == (n_query+1):
            heads = heads_all[:, :-1,:]   #256, 100, 64
            heads_y = heads_all[:,-1:,:]  #256, 1, 64
            return heads, heads_y #256, 100, 64  #256, 1, 64
        else:
            # print(heads_all.shape)
            return heads_all
        ###########################################
    def softmax_real(self, input, attn_mask=None):
        # if real:
        # real = torch.real(input)
        # else:
        real = 10*torch.cos(input.angle())
        if attn_mask is not None:
            real += attn_mask.unsqueeze(0).real.to(self.device)
        # abso[abso == float('inf')] = -abso[abso == float('inf')]
        return softmax(real, dim=-1).type(torch.complex64) ############
    
class Transformer_Encoder(nn.Module):
    def __init__(self, embed_dim, A_dim=64, node_dim=None, num_heads=4, device='cuda'):
        super(Transformer_Encoder, self).__init__()
        # self.relu_dropout = relu_dropout
        self.init_embed = ComplexLinear(node_dim, A_dim) if node_dim is not None else None
        self.attention = attention(input_dim=A_dim, embed_dim=embed_dim, num_heads=num_heads, Glimpse=None)
        self.FeedforwardNN = FeeaforwardNN(embed_dim)
        self.layer_norms = nn.ModuleList([Complex_LayerNorm(embed_dim) for _ in range(2)])
        self.device = device
    def forward(self, A, x):
        #A: 256, 100, 12
        #x: 256, 1, 64
        # B, C, L = x.shape 
        h = self.init_embed(A.view(-1, A.size(-1))).view(*A.size()[:2], -1) if self.init_embed is not None else A #256*100, 12->64  256, 100, 64
        residual_x = x #256, 1, 64
        residual_h = h #256, 100, 64
        h, x = self.attention(h, x) #256, 100, 64 #256, 1, 64
        x += residual_x 
        h += residual_h 
        h = self.layer_norms[0](h)
        x = self.layer_norms[0](x)
        h = self.FeedforwardNN(h) + h 
        x = self.FeedforwardNN(x) + x 
        h = self.layer_norms[1](h)
        x = self.layer_norms[1](x)
        return h, x  # (batch_size, graph_size, embed_dim) # (batch_size, 1, y_dim)  #256, 100, 64  #256, 1, 64

class Logit(nn.Module):
    def __init__(
            self,
            q_dim,
            k_dim,
            embed_dim
    ):
        super(Logit, self).__init__()
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.embed_dim = embed_dim
        self.pro_q = ComplexLinear(self.embed_dim, self.embed_dim)
        self.pro_k = ComplexLinear(self.embed_dim, self.embed_dim)
        self.pro_v = ComplexLinear(self.embed_dim, self.embed_dim)

        # self.dense = nn.Linear(20, 20)

        self.scaling = embed_dim ** -0.5  #  8 See Attention is all you need

    def forward(self, q, h=None, mask=None):
     
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, embed_dim)
        B, graph_size, k_dim = h.size() #  B, 20, 64
        n_query = q.size(1)
        q_dim = q.size(2)
        assert q.size(0) == B   
        assert q_dim == self.q_dim, "Wrong embedding dimension of q"
        assert k_dim == self.k_dim, "Wrong embedding dimension of h"

       
        Q = self.pro_q(q).view(B, 1, 1, -1).contiguous().permute(2, 0, 1, 3) # 1, B, 1, 64
        K = self.pro_k(h).unsqueeze(dim=0)  #1, B, 20, 64
   

        # Calculate compatibility (1, batch_size, n_query, graph_size)
        attn = self.scaling * (torch.matmul(Q, torch.conj_physical(K).transpose(-1, -2)))
        compatibility = 10*torch.tanh(attn.abs()) * torch.exp(-1j*attn.angle()) #1, B, 1, 20
        # compatibility = relu(attn.abs()) * torch.exp(-1j*attn.angle())
        # compatibility = attn.abs() #1, B, 1, 20

        out = self.C2R(compatibility.squeeze(2).squeeze(0)) #B, 20

        return out
    

    def C2R(self, input):
        power = torch.square(sigmoid(input.real)) + torch.square(sigmoid(input.imag))
        clipped_power = torch.clip(power,0, 1)

        return clipped_power


class Complex_Out(nn.Module):
    def __init__( self, num_heads, embed_dim, y_dim, x_dim, device='cuda'):
        super(Complex_Out, self).__init__()
        self.glimpse = attention(
            num_heads=num_heads, input_dim=x_dim, y_dim=y_dim, embed_dim=embed_dim, Glimpse=1)
        self.logit = Logit(q_dim=y_dim, k_dim=x_dim, embed_dim=embed_dim) 

    def forward(self, x, y):
        #x: 256, 100, 64  #y: 256, 1, 64
        y = self.glimpse(x,y) 
        out = self.logit(y,x)
        return out       
     
class Transformer_Decoder(nn.Module):
    def __init__(self, embed_dim, device='cuda'):
        super().__init__()
        self.conv = ComplexConv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1)
        self.layer_norms = nn.ModuleList([Complex_LayerNorm(embed_dim) for _ in range(2)])
        self.device = device
    def forward(self, x):
        x = self.conv(x) # 128, 2, 1, 72
        residual = x # 128, 2, 1, 72
        x = self.conv(x) # 128, 2, 1, 72
        x = complex_relu(x)# 128, 2, 1, 72
        x = self.conv(x)# 128, 2, 1, 72
        x += residual # 128, 2, 1, 72
        x = self.layer_norms[0](x) # 128, 2, 1, 72
        return x
    

if __name__=='__main__':
    encoder = Complex_Transformer().cuda()
     #A: B,C,L = 256, 20, 8 
        #x: B,L = 256, 64
        #output: 256, 20
    A = torch.tensor(torch.rand((256, 100, 12), dtype=torch.complex64)).cuda() #256, 100, 64  #256, 1, 64
    x = torch.tensor(torch.rand((256, 1, 64), dtype=torch.complex64)).cuda()
    # print(10*torch.tanh(A))
    out = encoder(A, x)
    print(out.shape, out)
