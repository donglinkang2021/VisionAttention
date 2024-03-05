import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Tuple, Optional
from einops import rearrange

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        raise NotImplementedError
    
def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
    ) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class CNN(BaseModel):
    def __init__(self, in_channels: int, n_channels: int, n_classes: int):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                n_channels, 
                kernel_size = 7, 
                stride = 2, 
                padding = 3           
            ),
            nn.BatchNorm2d(n_channels), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.res1 = BasicBlock(n_channels, 2 * n_channels, stride=2)
        self.res2 = BasicBlock(2 * n_channels, 4 * n_channels, stride=2)
        self.res3 = BasicBlock(4 * n_channels, 8 * n_channels, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * n_channels, n_classes)
        self.apply(self._init_weights)
        print(f"number of parameters: {self.get_num_params()/1e6:.6f} M ")

    def forward(self, x):
        x = self.downsample(x)  # B, 3, 32, 32  -> B, 64, 16, 16
        x = self.res1(x)        # B, 64, 16, 16 -> B, 128, 8, 8
        x = self.res2(x)        # B, 128, 8, 8  -> B, 256, 4, 4
        x = self.res3(x)        # B, 256, 4, 4  -> B, 512, 2, 2
        x = self.avgpool(x)     # B, 512, 2, 2  -> B, 512, 1, 1
        x = self.flatten(x)     # B, 512, 1, 1  -> B, 512
        x = self.fc(x)          # B, 512        -> B, n_classes
        return x
    
class ResHeads(BaseModel):
    def __init__(self, n_classes: int, n_head: int = None, pretrained_model: str='resnet18'):
        super().__init__()
        self.n_head = n_head
        if pretrained_model == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.classifier = nn.Linear(512, n_classes)
        elif pretrained_model == 'resnet34':
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            self.classifier = nn.Linear(512, n_classes)
        elif pretrained_model == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.classifier = nn.Linear(2048, n_classes)
        elif pretrained_model == 'resnet101':
            self.backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            self.classifier = nn.Linear(2048, n_classes)
        elif pretrained_model == 'resnet152':
            self.backbone = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
            self.classifier = nn.Linear(2048, n_classes)
        else:
            raise ValueError(f"Unknown model: {pretrained_model}")
        
        
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        print(f"number of parameters: {self.get_num_params()/1e6:.6f} M ")

    def freeze(self):
        for x in self.parameters():
            x.requires_grad = False
        for x in self.classifier.parameters():
            x.requires_grad = True

    def unfreeze(self):
        for x in self.parameters():
            x.requires_grad = True

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        if self.n_head is not None:
            x = rearrange(x, 'B (nh hs) -> B nh hs', nh=self.n_head)
            x = F.scaled_dot_product_attention(x, x, x)
            x = rearrange(x, 'B nh hs -> B (nh hs)')

        x = self.classifier(x)
        return x

class MLP(nn.Module):
    def __init__(self, n_embd, dropout, bias=True):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CausalSelfAttention(nn.Module):
    """mix the head and the multi-head attention together"""

    def __init__(self, n_embd: int, n_head: int, dropout: float, is_causal=False):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout = dropout
        self.is_causal = is_causal
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # project the queries, keys and values
        q, k, v = self.c_attn(x).split(C, dim=2)
        k = rearrange(k, 'B T (nh hs) -> B nh T hs', nh=self.n_head)
        q = rearrange(q, 'B T (nh hs) -> B nh T hs', nh=self.n_head)
        v = rearrange(v, 'B T (nh hs) -> B nh T hs', nh=self.n_head)

        # casual self-attention: ignore "future" keys during attention
        # masked attention
        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, 
            dropout_p=self.dropout if self.training else 0,
            is_causal=self.is_causal
        )
        
        # re-assemble all head outputs side by side
        y = rearrange(y, 'B nh T hs -> B T (nh hs)')
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd: int, n_head: int, dropout: float):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        assert n_embd % n_head == 0, 'n_embd must be divisible by n_head'
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.mlp = MLP(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
class ViHeads(BaseModel):
    """ViHeads is not ViT"""
    def __init__(
            self, 
            in_channels: int, 
            n_classes: int,
            image_size: int, 
            patch_size: int, 
            n_embd: int, 
            n_head: int, 
            n_layer: int,
            dropout: float 
        ):
        super().__init__()
        assert image_size % patch_size == 0, 'image size must be divisible by patch size'
        self.block_size = (image_size // patch_size) ** 2                  # number of patches H*W
        self.patch_flatten_dim = in_channels * patch_size ** 2             # number of elements in a patch p1*p2*C
        self.patch_size = patch_size                                       # patch size p1=p2=p
        self.n_embd = n_embd                                               # embedding dimension
        self.n_head = n_head                                               # number of heads
        
        self.transformer = nn.ModuleDict(dict(
            pte = nn.Linear(self.patch_flatten_dim, n_embd, bias=False),   # patch to embedding
            ppe = nn.Embedding(self.block_size, n_embd),                   # position embedding  
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd)
        ))

        self.fc = nn.Linear(self.block_size * self.n_embd, n_classes)
        
        self.apply(self._init_weights)
        print(f"number of parameters: {self.get_num_params()/1e6:.6f} M ")

    def forward(self, x):
        device = x.device
        x = rearrange(x, 'B C (H p1) (W p2) -> B (H W) (p1 p2 C)', p1=self.patch_size, p2=self.patch_size)
        
        pos = torch.arange(0, self.block_size, dtype=torch.long, device=device) # shape (t)

        tok_emb = self.transformer.pte(x)   # token embeddings of shape (B, T, n_embd)
        pos_emb = self.transformer.ppe(pos) # position embeddings of shape (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # x = rearrange(x, 'B T (nh hs) -> B T nh hs', nh=self.n_head)
        # x = F.scaled_dot_product_attention(x, x, x)
        # x = rearrange(x, 'B T nh hs -> B (T nh hs)')

        x = rearrange(x, 'B T D -> B (T D)')
        x = self.fc(x)  # B (T nh hs) -> B n_classes
        
        return x