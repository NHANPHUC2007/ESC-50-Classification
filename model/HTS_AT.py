import torch
import torch.nn as nn

class LayerNorm(nn.Module):          
    def __init__(self, emb_dim, eps=1e-5):  
        super().__init__()

        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# Notation:
# C is the embedding dimension

class Window_MultiHeadAttention(nn.Module):
    def __init__(self, C, num_heads, drop_out, qvk_bias=False, window_sizes: tuple[int, int] = (7, 7)):
        super().__init__()
        
        assert C % num_heads == 0, 'C dimension must be divisible by num_heads'
        
        self.window_height, self.window_width = window_sizes
        self.window_size = self.window_height * self.window_width
        self.C = C
        self.num_heads = num_heads
        self.head_dim = C // num_heads
        
        self.W_query = nn.Linear(C, C, bias=qvk_bias)
        self.W_key = nn.Linear(C, C, bias=qvk_bias)
        self.W_value = nn.Linear(C, C, bias=qvk_bias)
        
        self.out_proj = nn.Linear(C, C)
        self.drop_out = nn.Dropout(drop_out)
        
    # Assume the x is (b, num_window, window_size, C)
    def forward(self, x, attn_mask=None):
        b, num_window, _, _  = x.shape
        
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        queries = queries.view(b * num_window, self.window_size, self.num_heads, self.head_dim)    
        keys = keys.view(b * num_window, self.window_size, self.num_heads, self.head_dim)
        values = values.view(b * num_window, self.window_size, self.num_heads, self.head_dim)
        
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Output: b * num_window, num_heads, window_size, window_size
        attn_scores = queries @ keys.transpose(2, 3)
        
        if attn_mask is not None:
            attn_scores = attn_scores.contiguous().view(
                b, num_window, self.num_heads, self.window_size, self.window_size
            )
            
            # Output: num_window, 1, window_size, window_size
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(2)
            attn_scores = attn_scores + attn_mask
            
            attn_scores = attn_scores.view(b * num_window, self.num_heads, self.window_size, self.window_size)
            
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        
        attn_weights = self.drop_out(attn_weights)
        
        context_vecs = attn_weights @ values
        
        context_vecs = context_vecs.transpose(1, 2)
        
        context_vecs = context_vecs.contiguous().view(
            b, num_window, self.window_size, self.C
        )
        
        context_vecs = self.out_proj(context_vecs)
        
        return context_vecs
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim)
        )
        
    def forward(self, x):
        return self.layers(x)

def window_partition(x, window_W):
    """
        Args:
            x: (b, height, width, C)
        Output:
            output: (b, num_window, window_W, window_W, C)  
    """
    B, H, W, C = x.shape
    output = x.view(B, H // window_W, window_W, W // window_W, window_W, C)
    output = output.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, window_W, window_W, C)
    return output

class SwinTransformerBlock(nn.Module):
    def __init__(self, C, input_resolution, window_W, window_shift, 
                 drop_out, num_heads, qkv_bias=False, mlp_ratio=4):
        
        super().__init__()
        
        self.C = C
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio
        
        if min(input_resolution) < window_W:
            window_W = min(input_resolution)
            window_shift = window_W // 2
        self.window_shift = window_shift
        self.window_W = window_W
        
        self.norm1 = LayerNorm(C)
        
        self.attn = Window_MultiHeadAttention(
            C, num_heads, drop_out, 
            qvk_bias=qkv_bias, 
            window_sizes=(window_W, window_W)
        )
        
        self.norm2 = LayerNorm(C)
        
        self.mlp = MLP(self.C, self.C * self.mlp_ratio)
        
        self.drop_out = nn.Dropout(drop_out)
        
        if self.window_shift > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            cnt = 0
            
            for h in (
                slice(0, -window_W),
                slice(-window_W, -window_shift),
                slice(-window_shift, None)
            ):
                for w in (
                    slice(0, -window_W),
                    slice(-window_W, -window_shift),
                    slice(-window_shift, None)
                ):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            window_mask = window_partition(img_mask, window_W).view(-1, window_W * window_W)
            window_mask = window_mask.unsqueeze(1) - window_mask.unsqueeze(2)
            attn_mask = window_mask.masked_fill(window_mask != 0, -torch.inf) \
                                   .masked_fill(window_mask == 0, float(0.0))
        else:
            attn_mask = None
            
        self.register_buffer('attn_mask', attn_mask)
        
    '''
        output: (batch, image_H * image_W, C)
    '''
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'The input resolution is incorrect'
        
        shortcut = x
        x = self.norm1(x)
        
        # Cyclic shift
        x = x.view(B, H, W, C)
        
        if self.window_shift > 0:
            shift_x = torch.roll(x, shifts=(-self.window_shift, -self.window_shift), dims=(1, 2))
        else:
            shift_x = x
            
        window_x = window_partition(shift_x, self.window_W)
        window_x = window_x.view(B, -1, self.window_W * self.window_W, C)
        
        # (b, num_window, window_size, C)
        attn_windows = self.attn(window_x, self.attn_mask)
        
        # (b, H // window_W, window_W, W // window_W, window_W, C)
        attn_windows = attn_windows.view(
            B, 
            H // self.window_W, W // self.window_W, 
            self.window_W, self.window_W,
            C
        )
        attn_windows = attn_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
        
        shift_x = attn_windows
        
        # Reverse cyclic shift 
        if self.window_shift > 0:
            x = torch.roll(shift_x, shifts=(self.window_shift, self.window_shift), dims=(1, 2))
        else:
            x = shift_x
        
        x = x.view(B, L, C)
        x = self.drop_out(x)
        x = x + shortcut
        
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_out(x)
        
        x = x + shortcut
        
        return x
  
class PatchMerge(nn.Module):
    def __init__(self, input_resolution, dim_in, dim_out=None):
        super().__init__()
        
        self.H, self.W = input_resolution
        self.dim_in = dim_in
        self.dim_out = dim_out if dim_out else 2 * dim_in
        
        self.norm = LayerNorm(4 * self.dim_in)
        self.reduction = nn.Linear(4 * dim_in, self.dim_out, bias=True)
        
    def forward(self, x):
        '''
            Input: (B, L, C)
        '''
        B, L, C = x.shape
        
        assert L == self.H * self.W, 'Input resolution mismatch'
        assert C == self.dim_in, 'Embedding dimmension mismatch'
        
        x = x.view(B, self.H, self.W, C)
        
        x = x.view(B, self.H // 2, 2, self.W // 2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.contiguous().view(B, self.H // 2, self.W // 2, 4 * C)
        
        x = self.norm(x) 
        x = self.reduction(x) # B, H//2, W//2, dim_out,  Reduce the size by 2 (potentially)
        x = x.view(B, -1, self.dim_out)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, input_resolution, patch_size, in_channels, emb_dim, patch_stride):
        super().__init__()
        
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.patch_stride = patch_stride
        self.patch_resolution = (self.input_resolution[0] // patch_stride, self.input_resolution[1] // patch_stride)
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=(patch_size, patch_size), stride=patch_stride)
        self.norm = LayerNorm(emb_dim)
        
    def forward(self, x):
        '''
            Args:
                x: (B, H, W)
            Return:
                (B, N, emb_dim)
        '''
        B, H, W = x.shape
        x = x.unsqueeze(1)
        
        x = self.proj(x) # B, out_channel, H, W
        x = x.flatten(2).transpose(1, 2)
        
        x = self.norm(x)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, C, input_resolution, window_W, num_blk,
                 drop_out, num_heads, qkv_bias=False, mlp_ratio=4, merge=False):
        super().__init__()
        
        self.C = C
        self.input_resolution = input_resolution
        self.window_W = window_W
        self.drop_out = drop_out
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.mlp_ratio = mlp_ratio
        self.num_blk = num_blk
        self.merge = merge
        
        self.trf_block = nn.Sequential(
            *[SwinTransformerBlock(
                C=C, 
                input_resolution=input_resolution,
                window_W=window_W,
                window_shift= 0 if (i % 2 == 0) else window_W //2,
                drop_out=drop_out,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                mlp_ratio=mlp_ratio)
               
              for i in range(num_blk)]
        )
        
        if merge:
            self.patch_merge = PatchMerge(input_resolution, C)
        
    def forward(self, x):
        '''
            Args:
                x: (B, L, C)
        '''
        H, W = self.input_resolution
        L = x.shape[1]
        assert L == H * W, 'The input resolution is incorrect'
        
        x = self.trf_block(x)
        if self.merge:
            x = self.patch_merge(x)
        return x

class HTSAT_Backbone(nn.Module):
    def __init__(self, C=96, input_resolution=(64, 512), window_W=8,
                 depth=[2, 2, 6, 2], drop_out=0.1, num_heads=[4, 8, 16, 32], 
                 qkv_bias=True, mlp_ratio=4, in_channels=1, patch_size=4, patch_stride=4,
                 d_feature=200):
        super().__init__()
        
        self.C = C
        self.input_resolution = input_resolution
        self.window_W = window_W
        self.depth = depth
        self.drop_out = drop_out
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.mlp_ratio = mlp_ratio
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        
        self.patch_embed = PatchEmbed(input_resolution=input_resolution, patch_size=patch_size, 
                                        patch_stride=patch_stride, emb_dim=C, in_channels=in_channels)
        
        num_patches = self.path_embed.num_patches
        patch_resolution = self.path_embed.patch_resolution
        len_depth = len(depth)
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.C))
        
        # Do the layers of transformer
        self.layers = nn.Sequential( *[
            TransformerLayer(
                C= (2**i) * C,
                input_resolution=(patch_resolution[0] // (2**i), patch_resolution[1] // (2**i)),
                window_W=window_W,
                num_blk=depth[i],
                num_heads=num_heads[i],
                drop_out=drop_out,
                qkv_bias=qkv_bias,
                mlp_ratio=mlp_ratio,
                merge=False if i == len_depth - 1 else True
            )
            for i in range(len_depth)
        ])

        # Feature Output
        reduce_ratio = len_depth - 1
        self.trf_final_resolution = (patch_resolution[0] // (2**reduce_ratio), patch_resolution[1] // (2**reduce_ratio))
        
        self.proj = nn.Conv2d(
            C * 2 ** reduce_ratio, 
            d_feature, 
            kernel_size=(self.trf_final_resolution[0], 3), 
            padding=(0, 1), stride=1
        )
        
        self.len_depth = len_depth
        self.final_norm = LayerNorm(C * 2 ** reduce_ratio)
    
    def forward(self, x):
        '''
            Args:
            x: (B, H, W)
            
            Return:
            x: (B, d_feature)
        '''
        B = x.shape[0]
                
        x = self.patch_embed(x)
        x = x + self.absolute_pos_embed
        
        x = self.layers(x) # B, F/8P * T/8P, 8C
        x = self.final_norm(x)
        
        x = x.transpose(1, 2) # B, 8C, F/8P * T/8P
        x = x.view(B, self.C * 2 ** (self.len_depth - 1), self.trf_final_resolution[0], self.trf_final_resolution[1])
    
        x = self.proj(x) # B, d_feature, 1, T/8P
        x = x.squeeze(2).transpose(1, 2) # B, T/8P, d_fearture
        
        x = x.mean(dim=1) # B, d_feature
        return x
        
class HTSAT_Model(nn.Module):
    def __init__(self, C=96, input_resolution=(64, 512), window_W=8,
                 depth=[2, 2, 6, 2], drop_out=0.1, num_heads=[4, 8, 16, 32], 
                 qkv_bias=True, mlp_ratio=4, in_channels=1, patch_size=4, patch_stride=4,
                 d_feature=200, output_dim=50):
        super().__init__()
        
        self.backbone = HTSAT_Backbone(
            C=C, input_resolution=input_resolution, window_W=window_W,
            depth=depth, drop_out=drop_out, num_heads=num_heads, qkv_bias=qkv_bias,
            mlp_ratio=mlp_ratio, in_channels=in_channels, patch_size=patch_size,
            patch_stride=patch_stride, d_feature=d_feature
        )
        
        # Output
        self.classifier = nn.Linear(d_feature, output_dim, bias=True)
    
    def forward_features(self, x):
        x = self.backbone(x)
        
        return x
     
    def forward(self, x):
        '''
            Args:
            x: (B, H, W)
            
            Return:
            x: (B, output_dim)
        '''
        feature = self.forward_features(x)        
        x = self.classifier(feature) # B, output_dim
        return x


if __name__ == '__main__':
  pass  