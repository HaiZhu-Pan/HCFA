import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init

from MyNetwork.dynamic_conv import Dynamic_conv2d, Dynamic_conv3d


##dynamic_conv
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return Dynamic_conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv3x3_3d(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return Dynamic_conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                   dilation=dilation)

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim,eps=1e-6)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        # x = x*0.5 + x.mean(dim=1, keepdim=True)
        # x= (x + x.mean(dim=1, keepdim=True)) * 0.5  ####new
        return x


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.wq = nn.Linear(dim//heads, dim)
        self.wk = nn.Linear(dim//heads, dim)
        self.wv = nn.Linear(dim//heads, dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        # torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        # torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim * heads, dim)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, c, h = *x.shape, self.heads
        # qkv = self.to_qkv(x).chunk(3, dim=-1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions
        q = self.wq(x[:, 0:1, ...].reshape(b, 1, h, c // h)).permute(0, 2, 1,3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x.reshape(b, n, h, c // h)).permute(0, 2, 1,3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x.reshape(b, n, h, c // h)).permute(0, 2, 1,3)  # BNC -> BNH(C/H) -> BHN(C/H)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask
        #
        # m_r = torch.ones_like(dots) * 0.1
        # dots = dots + torch.bernoulli(m_r) * -1e12
        #
        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out




class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x

class LxNet(nn.Module):
    def __init__(
            self,
            num_classes=6,
            num_tokens= 4 ,######实验
            depth= 2 ,
            dim=64,
            mlp_dim=32,
            heads= 4 ,######实验
            h_in_dim=64,
            l_in_dim=1,
            dropout=0.1,
            emb_dropout=0.1,

    ):
        super(LxNet, self).__init__()
        self.name = 'LXNet'
        self.L = num_tokens
        self.cT = dim

        # Tokenization
        self.token_wA1 =nn.Parameter(torch.empty(1, 1, dim),  # self.L = num_tokens
                                     requires_grad=True)
        self.token_wA = nn.Parameter(torch.empty(1, self.L, dim),  # self.L = num_tokens
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)  # Glorot初始化 正态分布  尽可能的让输入和输出服从相同的分布，这样就能够避免后面层的激活函数的输出值趋向于0。
        torch.nn.init.xavier_normal_(self.token_wA1)
        self.token_wV1 = nn.Parameter(torch.empty(1, 1, self.cT),  # self.cT = dim
                                     requires_grad=True)
        self.token_wV = nn.Parameter(torch.empty(1, dim, self.cT),  # self.cT = dim
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)
        torch.nn.init.xavier_normal_(self.token_wV1)
        self.pos_embedding = nn.Parameter(torch.empty(1, num_tokens + 1, dim))  # 加了 cls token  所以num_tokens+1
        torch.nn.init.normal_(self.pos_embedding, std=.02)
        self.pos_embedding1 = nn.Parameter(torch.randn(1, num_tokens +1, dim))  # 有创
        # self.pos_embedding1 = nn.Parameter(torch.randn(1, num_tokens , dim))  # 无创
        torch.nn.init.normal_(self.pos_embedding1, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.mlp_head = nn.Sequential(nn.LayerNorm((num_tokens+1)*dim), nn.Linear((num_tokens+1)*dim, num_classes))
        self.out = nn.Linear(dim, num_classes)
        # self.conv0 = nn.Sequential(
        #     conv3x3(64, 1),
        #     nn.BatchNorm2d(1),
        #     nn.ReLU())
        self.conv1 = nn.Sequential(conv3x3(32,64),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(conv3x3_3d(1,8),
                                   nn.BatchNorm3d(8),
                                   nn.ReLU())
        self.conv3  = nn.Sequential(
            conv3x3(8 * h_in_dim, 64),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.conv4  = nn.Sequential(
            conv3x3(l_in_dim, 16),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.conv5 = nn.Sequential(
            conv3x3(16, 64),
            nn.BatchNorm2d(64),
            nn.ReLU())

        ##动态卷积消融
        # self.conv2b = nn.Sequential(nn.Conv3d(1, 8,(3,3,3),padding=1),
        #                            nn.BatchNorm3d(8),
        #                            nn.ReLU())
        # self.conv3b = nn.Sequential(
        #     nn.Conv2d(8 * h_in_dim, 64,(3,3)),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU())
        # self.conv4b = nn.Sequential(
        #     nn.Conv2d(l_in_dim, 16,(3,3)),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU())
        # self.conv5b = nn.Sequential(
        #     nn.Conv2d(16, 64,(3,3)),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU())
        ##动态卷积消融 end

        self.head =  nn.Sequential(
            nn.Linear(dim, num_classes)
        )
        self.weight_lambda = nn.Parameter(torch.ones(2) / 2, requires_grad=True)
        self.weight_lambda1 = nn.Parameter(torch.ones(2) , requires_grad=True)
        self.collector = None

    def forward(self, x1, x2, mask=None):
        #hsi
        x1 = self.conv2(x1)  #

        x1 = rearrange(x1, 'b c h w y ->b (c y) h w')  # 64 224 9 9    c*d
        x1 = self.conv3(x1) # 64 64 7 7
        t1 = x1
        x1 = rearrange(x1, 'b c h w -> b (h w) c')  # 64  49 64

        # x0 = t1
        # x0 = self.conv0(x0)
        # x0 = rearrange(x0, 'b c h w -> b (h w) c')
        #lidar
        # x2 = x2.squeeze(dim=1)
        x2 = rearrange(x2, 'b c h w y ->b (c y) h w')
        x2 = self.conv4(x2)
        x2 = self.conv5(x2)
        # x2 = self.maxpool2d(x2)
        t2 = x2
        x2 = rearrange(x2, 'b c h w -> b (h w) c')  # 64 81 64

        #Classification1
        # h = t1.mean([-2, -1])
        # cls_out1 = t1.mean([-2, -1])
        # cls_out2 = t2.mean([-2, -1])
        # out = torch.cat((cls_out1,cls_out2), dim=1)
        # out = self.head(out)
        cls_out1 = self.head(t1.mean([-2, -1]))
        cls_out2 = self.head(t2.mean([-2, -1]))
        out = cls_out1 + cls_out2

        #tokeniza
        wa0 = rearrange(self.token_wA1, 'b h w -> b w h') #1 64 1

        A0 = torch.einsum('bij,bjk->bik', x1, wa0)  # 1 49 1
        A0 = rearrange(A0, 'b h w -> b w h')  # 1 1 49
        A0 = A0.softmax(dim=-1) #1 1 49

        VV0 = torch.einsum('bij,bjk->bik', x1, self.token_wV1)  # 1 49 64
        T0 = torch.einsum('bij,bjk->bik', A0, VV0)  # 1 1 64

        wa1 = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose  1 64 4 wa1:1 64 4
        # 爱因斯坦求和约定：用于简洁的表示乘积、点积、转置等方法。
        A1 = torch.einsum('bij,bjk->bik', x1, wa1)  # 1 49 4
        A1 = rearrange(A1, 'b h w -> b w h')  # Transpose 64 4 49
        A1 = A1.softmax(dim=-1)

        VV1 = torch.einsum('bij,bjk->bik', x1, self.token_wV)  # 1 49 64
        T1 = torch.einsum('bij,bjk->bik', A1, VV1)  # 64 4 64

        wa2 = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        A2 = torch.einsum('bij,bjk->bik', x2, wa2)
        A2 = rearrange(A2, 'b h w -> b w h')  # Transpose
        A2 = A2.softmax(dim=-1)

        VV2 = torch.einsum('bij,bjk->bik', x2, self.token_wV)
        T2 = torch.einsum('bij,bjk->bik', A2, VV2)

        wa3 = rearrange(self.token_wA1, 'b h w -> b w h')  # Transpose
        A3 = torch.einsum('bij,bjk->bik', x2, wa3)
        A3 = rearrange(A3, 'b h w -> b w h')  # Transpose
        A3 = A3.softmax(dim=-1)

        VV3 = torch.einsum('bij,bjk->bik', x2, self.token_wV1)
        T3 = torch.einsum('bij,bjk->bik', A3, VV3)

        wa4 = rearrange(self.token_wA, 'b h w -> b w h')
        A4 = torch.einsum('bij,bjk->bik', x1, wa4)  # 64 49 4
        A4 = rearrange(A4, 'b h w -> b w h')
        A4 = A4.softmax(dim=-1)

        VV4 = torch.einsum('bij,bjk->bik', x1, self.token_wV)  # 64 49 64
        T4 = torch.einsum('bij,bjk->bik', A4, VV4)  # 64 4 64a


        # x2= T2 #无创
        x2 = torch.cat((T0, T2), dim=1)  #有创  T0 1,64  T2 4,64
        # cls_tokens2 = self.cls_token.expand(x2.shape[0], -1, -1)
        # x2 = torch.cat((cls_tokens2,x2), dim=1)
        x2 += self.pos_embedding1
        x2 = self.dropout(x2)
        # x2 = self.transformer(x2, mask)  # main game
        # x2 = self.to_cls_token(x2[:, 0])
        # x2 = self.nn1(x2)

        # x3 = T4  #无创
        x3 = torch.cat((T3, T4), dim=1)  #有创 T3 1,64  T4 4,64
        # cls_tokens3 = self.cls_token.expand(x3.shape[0], -1, -1)  # 64 1 64
        # x3 = torch.cat((cls_tokens3, x3), dim=1)
        x3 += self.pos_embedding1
        x3 = self.dropout(x3)

        #fusion
        x2 = self.transformer(x2)
        x3 = self.transformer(x3)
        # x2 = self.encoder_norm(x2)
        # x3 = self.encoder_norm(x3)
        x2, x3 = map(lambda t: t[:, 0], (x2, x3))
#         x2,x3 = self.fusion_encoder(x2, x3)
#         x2 =  x2.mean(dim=1)
#         x3 = x3.mean(dim=1)
#         x2 = x2.reshape(x2.shape[0], -1)
#         x3 = x3.reshape(x3.shape[0], -1)
#         x2, x3 = map(lambda t: t[:, 0], (x2, x3))
        x = self.out(x2) + self.out(x3)
        # x = self.mlp_head(x2) * self.weight_lambda[0] + self.mlp_head(x3) * self.weight_lambda[1]
        # x = self.nn1(x)
        weight = F.softmax(self.weight_lambda, 0)
        weight1 = F.softmax(self.weight_lambda1, 0)
        # print("out", out)
        # print("x", x)
        t= weight[0]*x+weight[1]*out
        # t = x
        # t = 0.5741 * x + 0.4259 * out
#         t = self.weight_lambda1[0] * x + self.weight_lambda1[1] * out
        self.collector = t
        # return weight1,out,t
        return weight,weight1,out,t

if __name__ == '__main__':
    model = LxNet(num_classes=11,h_in_dim=63,l_in_dim=1).to("cuda")
    print(model)
    input1 = torch.randn(64,1, 7, 7, 63).to("cuda")
    input2 = torch.randn(64,1, 7, 7, 1).to("cuda")
    x = model(input1,input2)
    # _,out,x = model(input1, input2)
    # print(out.size(),x.size())
    # summary(model, [( 1,7, 7,63), (1,7, 7,1)],device='cuda')


