import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):

    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=768):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def get_num_patches(self):
        return self.n_patches

    def forward(self, x):

        # x: (B, 3, W, H) -> (B, embed_dim, sqrt(n_patches), sqrt(n_patches))
        x = self.proj(x)

        # x: (B, embed_dim, sqrt(n_patches), sqrt(n_patches)) -> (B, embed_dim, n_patches)
        x = x.flatten(2)  # flattens starting from dim 2

        # x: (B, embed_dim, n_patches) -> (B, n_patches, embed_dim)
        x = x.transpose(1, 2)

        return x


class FFN(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, dropout_p):

        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout_p)

    def forward(self, x):

        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Attention(nn.Module):

    def __init__(self, embed_dim, n_heads=12, qkv_bias=True, attn_p=0.0, proj_p=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.dim = embed_dim

        assert (
            embed_dim % n_heads == 0
        ), "Embedding dimension must be divisible by number of heads"

        self.head_dim = embed_dim // n_heads

        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x, return_attn_weights=True):

        # B: batch size
        # P: num of patches
        # D: dimension (embed_dim)
        B, P, D = x.shape

        qkv = self.qkv(x)

        # x: B, P, 3*embed_dim -> x: B, P, 3, n_heads, head_dim
        qkv = qkv.view(B, P, 3, self.n_heads, self.head_dim)

        # x: B, P, 3, n_heads, head_dim -> 3, B, n_heads, P, head_dim
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # split qkv weights across first dim
        wq, wk, wv = torch.unbind(qkv, dim=0)

        # prepare wk for scaled dot product attention
        wk = wk.transpose(-2, -1)

        # attn: B, n_heads, P, P
        attn = (wq @ wk) * self.scale

        # apply softmax
        attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        # weighted_avg: B, n_heads, P, head_dim
        weighted_avg = attn @ wv

        # return to original dimension
        weighted_avg = weighted_avg.transpose(1, 2).contiguous().view(B, P, D)

        x = self.proj(weighted_avg)

        x = self.proj_drop(x)

        if return_attn_weights:
            return x, weighted_avg
        else:
            return x, None


class VITBlock(nn.Module):

    def __init__(
        self,
        embed_dim,
        n_heads,
        scale_factor=4,
        qkv_bias=True,
        attn_p=0.0,
        proj_p=0.0,
        eps=1e-6,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim, eps=eps)

        self.attn = Attention(embed_dim, n_heads, qkv_bias, attn_p, proj_p)

        self.norm2 = nn.LayerNorm(embed_dim, eps=eps)
        self.mlp = FFN(
            in_features=embed_dim,
            hidden_features=embed_dim * scale_factor,
            out_features=embed_dim,
            dropout_p=proj_p,
        )

    def forward(self, x, return_attn_weights=True):

        x_initial = x

        x, attn_weights = self.attn(
            self.norm1(x), return_attn_weights=return_attn_weights
        )

        x = x_initial + x

        x_initial = x

        x = self.mlp(self.norm2(x))

        x = x_initial + x

        return x, attn_weights


class VIT(nn.Module):

    def __init__(
        self,
        img_size,
        patch_size,
        in_channels=3,
        embed_dim=768,
        n_classes=1000,
        depth=12,
        n_heads=12,
        scale_factor=4,
        qkv_bias=True,
        attn_p=0.0,
        proj_p=0.0,
        eps=1e-6,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # create classification, similar to BERT
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # create positional embedding, note this is learnt, unlike original transformer implementation which utilized sinusoidal encoding.
        # we concatenate the cls token to the patches, hence 1 + n_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.get_num_patches(), embed_dim)
        )

        # dropout for pos embedding
        self.pos_drop = nn.Dropout(proj_p)

        # our transformer blocks
        self.blocks = nn.ModuleList(
            [
                VITBlock(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    scale_factor=scale_factor,
                    qkv_bias=qkv_bias,
                    attn_p=attn_p,
                    proj_p=proj_p,
                    eps=eps,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=eps)

        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x, return_attn_weights=True):

        B = x.shape[0]

        x = self.patch_embed(x)

        # this expands first dimension of classification token to that of batch size
        cls_token = self.cls_token.expand(B, -1, -1)

        # concatenate cls token along patch dimension
        x = torch.cat((cls_token, x), dim=1)

        # add positional embedding
        x = x + self.pos_embed

        x = self.pos_drop(x)

        # store attention weights from each block
        all_attn_weights = []
        for block in self.blocks:
            x, attn_w = block(x, return_attn_weights=return_attn_weights)
            all_attn_weights.append(attn_w)

        x = self.norm(x)
        prepare_for_classification = x[:, 0]  # classification token only

        x = self.head(prepare_for_classification)

        return x
