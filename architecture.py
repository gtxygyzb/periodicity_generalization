import torch
from torch import nn
import torch.nn.functional as F
import math
from transformers import AutoModel
from mamba_ssm.models.mixer_seq_simple import MixerModel
import os, math, gc, importlib
# Define a model registry
model_registry = {}
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from torch.utils.cpp_extension import load

# Create a decorator to register models
def register_model(model_name):
    def decorator(cls):
        model_registry[model_name] = cls
        return cls
    return decorator

# Define a function to retrieve and instantiate the model class by model_name
def get_model_by_name(model_name, *args, **kwargs):
    model_cls = model_registry.get(model_name)
    if model_cls is None:
        raise ValueError(f"No model found with model_name{model_name}.")
    return model_cls(*args, **kwargs)

# Use the decorator to register the model class

@register_model('FANLayer')
class FANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(FANLayer, self).__init__()
        self.input_linear_p = nn.Linear(input_dim, output_dim//4, bias=bias) # There is almost no difference between bias and non-bias in our experiments.
        self.input_linear_g = nn.Linear(input_dim, (output_dim-output_dim//2))
        self.activation = nn.GELU()        
    
    def forward(self, src):
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)
        
        output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        return output
    
@register_model('FANLayerGated')
class FANLayerGated(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, gated = True):
        super(FANLayerGated, self).__init__()
        self.input_linear_p = nn.Linear(input_dim, output_dim//4, bias=bias) 
        self.input_linear_g = nn.Linear(input_dim, (output_dim-output_dim//2))
        self.activation = nn.GELU()        
        if gated:
            self.gate = nn.Parameter(torch.randn(1, dtype=torch.float32))
    
    def forward(self, src):
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)
        
        if not hasattr(self, 'gate'):
            output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        else:
            gate = torch.sigmoid(self.gate)
            output = torch.cat((gate*torch.cos(p), gate*torch.sin(p), (1-gate)*g), dim=-1)
        return output

@register_model('FAN')
class FAN(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3):
        super(FAN, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)   
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(FANLayer(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, src):
        output = self.embedding(src)
        for layer in self.layers:
            output = layer(output)
        return output

@register_model('FANGated')
class FANGated(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3, gated = True):
        super(FANGated, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)   
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(FANLayerGated(hidden_dim, hidden_dim, gated = gated))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, src):
        output = self.embedding(src)
        for layer in self.layers:
            output = layer(output)
        return output

@register_model('MLP')
class MLPModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=2048, num_layers=3, use_embedding=True):
        super(MLPModel, self).__init__()
        self.activation = nn.GELU()  
        self.layers = nn.ModuleList() 
        if use_embedding:
            self.embedding = nn.Linear(input_dim, hidden_dim)
            self.layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activation])
        else:
            self.layers.extend([nn.Linear(input_dim, hidden_dim), self.activation])
        
        for _ in range(num_layers - 2):
            self.layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activation])
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, src):
        output = self.embedding(src) if hasattr(self, 'embedding') else src
        for layer in self.layers:
            output = layer(output)
        return output

class SinPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return x

class RoPEPositionalEncoding(torch.nn.Module):
    """
    Rotary Positional Encoding (RoPE)
    真正的RoPE实现，通过旋转矩阵注入位置信息
    """
    
    def __init__(self, dim: int, max_len: int = 5000, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base
        
        # 预计算theta值
        theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('theta', theta)
        
    def forward(self, x: torch.Tensor, start_pos: int = 0):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, hidden_dim]``
            start_pos: 起始位置，用于推理时的增量生成
        """
        seq_len, batch_size, hidden_dim = x.shape
        assert hidden_dim == self.dim, f"Hidden dimension {hidden_dim} must match RoPE dimension {self.dim}"
        
        # 创建位置索引
        positions = torch.arange(start_pos, start_pos + seq_len, device=x.device).float()
        
        # 计算角度：position * theta
        angles = positions.unsqueeze(-1) * self.theta.unsqueeze(0)  # [seq_len, dim//2]
        
        # 计算cos和sin
        cos_vals = torch.cos(angles)  # [seq_len, dim//2]
        sin_vals = torch.sin(angles)  # [seq_len, dim//2]
        
        # 将x分成两部分：前一半和后一半
        x1 = x[..., : self.dim // 2]  # [seq_len, batch_size, dim//2]
        x2 = x[..., self.dim // 2 :]  # [seq_len, batch_size, dim//2]
        
        # 应用旋转操作
        # [-x2, x1] * [cos, sin] + [x1, x2] * [cos, -sin] 的简化形式
        rotated_x1 = x1 * cos_vals.unsqueeze(1) - x2 * sin_vals.unsqueeze(1)
        rotated_x2 = x1 * sin_vals.unsqueeze(1) + x2 * cos_vals.unsqueeze(1)
        
        # 合并结果
        result = torch.cat([rotated_x1, rotated_x2], dim=-1)
        return result
        


@register_model('Transformer')
class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, num_layers=5, 
                 pretrained_name="/home/gtxygyzb/models/Qwen/Qwen2.5-0.5B",
                 freeze_emb=True,
                 num_heads=12, norm_first = True, encoder_only=True, decoder_only=False):
        #super(TransformerModel, self).__init__()
        super().__init__()

        # 加载预训练模型
        base_model = AutoModel.from_pretrained(pretrained_name)
        # 用预训练 embedding
        self.embedding = base_model.get_input_embeddings()
        self.hidden_dim = self.embedding.embedding_dim
        hidden_dim = self.hidden_dim
        print("hidden_dim:", self.hidden_dim)

        if freeze_emb:
            for p in self.embedding.parameters():
                p.requires_grad = False

        self.pos_encoder = RoPEPositionalEncoding(hidden_dim)
        self.encoder_only = encoder_only
        self.decoder_only = decoder_only
        assert not (self.encoder_only and self.decoder_only)
        if self.encoder_only:
            encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, norm_first = norm_first)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        elif self.decoder_only:
            decoder_layers = nn.TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim, norm_first = norm_first)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        else:
            encoder_layers = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, norm_first = norm_first)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers//2)
            decoder_layers = nn.TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim, norm_first = norm_first)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers//2)
        self.out = nn.Linear(self.hidden_dim, output_dim)

        
    def forward(self, src):
        # src: (batch, seq_len)
        src = self.embedding(src)
        src = src.transpose(0, 1)  # (seq_len, batch, hidden)
        src = self.pos_encoder(src)
    
        # ====== 新增：causal attention mask ======
        seq_len = src.size(0)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=src.device),
            diagonal=1
        ).bool()
        # ========================================
    
        if self.encoder_only:
            src = self.transformer_encoder(src, mask=causal_mask)
        elif self.decoder_only:
            src = self.transformer_decoder(src, src)
        else:
            src = self.transformer_encoder(src)
            src = self.transformer_decoder(src, src)
    
        logits = self.out(src).transpose(0, 1)
        return logits



class PartialLayerNorm(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.half_dim = hidden_dim // 2
        self.ln = nn.LayerNorm(self.half_dim)
        # self.dy_ln = nn.LayerNorm(hidden_dim - self.half_dim)

    def forward(self, x):
        # 假设 x 形状是 (seq_len, batch_size, hidden_dim) 或 (batch_size, seq_len, hidden_dim)
        x1 = x[..., :self.half_dim]  # 前一半做LayerNorm
        x2 = x[..., self.half_dim:]  # 后一半不变
        x1 = self.ln(x1)
        # x2 = self.dy_ln(x2)
        return torch.cat([x1, x2], dim=-1)

@register_model('FFTLayer')
class FFTLayer(nn.Module):
    def __init__(self, hidden_dim, bias=True):
        super().__init__()
        self.hidden_half = hidden_dim // 2

        # 原始时域部分
        self.linear_orig = nn.Linear(hidden_dim, self.hidden_half, bias=bias)

        # 频域线性映射
        self.linear_freq = nn.Linear(hidden_dim, self.hidden_half, bias=bias)

        # 输出投影
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.activation = nn.GELU()

        # MLP: 频率 -> 增益（连续频率建模）
        self.mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, src):
        # src: [L, B, D]
        seq_len, batch, hidden_dim = src.shape
        src = src.transpose(0, 1)  # [B, L, D]

        # 1️⃣ 时域部分
        orig_part = self.activation(self.linear_orig(src))  # [B, L, D/2]

        # 2️⃣ 频域部分
        freq_in = self.linear_freq(src)  # [B, L, D/2]
        X_freq = torch.fft.rfft(freq_in, dim=1)  # [B, F, D/2]

        # 连续频率索引
        F = X_freq.shape[1]
        omega = (2 * torch.pi * torch.arange(F, device=src.device) / seq_len).unsqueeze(-1)  # [F,1]
        gain = self.mlp(omega).squeeze(-1)  # [F]
        gain = gain.unsqueeze(0).unsqueeze(-1)  # [1,F,1]
        X_freq = X_freq * gain  # 连续频率加权

        # 反变换
        freq_part = torch.fft.irfft(X_freq, n=seq_len, dim=1)  # [B,L,D/2]

        # 3️⃣ 拼接 + 投影
        x_out = torch.cat([orig_part, freq_part], dim=-1)
        x_out = self.proj(x_out)
        return x_out.transpose(0, 1)  # [L,B,D]



##################################################################################

class FANformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, norm_first=True, dropout=0.1):
        super().__init__()
        self.norm_first = norm_first
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # self.fft_layer = FFTLayer(hidden_dim, hidden_dim)
        self.fan_layer = FANLayer(hidden_dim, hidden_dim)

        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # prenorm
        h = self.attn_norm(x) if self.norm_first else x
        # h = x

        # FANLayer
        h = self.fan_layer(h)
        # FFTLayer
        # h = self.fft_layer(h)

        # MultiheadAttention
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + self.dropout1(attn_out)

        # FFN
        h2 = self.ff_norm(x) if self.norm_first else x
        # h2 = x
        ff_out = self.ff(h2)
        x = x + self.dropout2(ff_out)
        return x


@register_model('FANformer')
class FANformerModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=768, num_layers=12,
                 num_heads=12, norm_first=True, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = RoPEPositionalEncoding(hidden_dim)

        # FANformer 层
        self.layers = nn.ModuleList([
            FANformerLayer(hidden_dim, num_heads, norm_first, dropout)
            for _ in range(num_layers)
        ])

        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, src):
        x = self.embedding(src).unsqueeze(0)
        x = self.pos_encoder(x) # x: (seq_len, batch, hidden_dim)
        for layer in self.layers:
            x = layer(x)

        return self.out(x)


@register_model('Qwen2.5Embedding-FANformer')
class Qwen_FANformerModel(nn.Module):
    def __init__(self, pretrained_name="/home/gtxygyzb/models/Qwen/Qwen2.5-0.5B", output_dim=1,
                 num_layers=5, num_heads=12, norm_first=True, dropout=0.1, freeze_emb=True):
        super().__init__()
        # 加载预训练模型
        base_model = AutoModel.from_pretrained(pretrained_name)

        # 用预训练 embedding
        self.embedding = base_model.get_input_embeddings()
        self.hidden_dim = self.embedding.embedding_dim
        # 压缩和展开层
        
        print("hidden_dim:", self.hidden_dim)

        if freeze_emb:
            for p in self.embedding.parameters():
                p.requires_grad = False

        self.pos_encoder = RoPEPositionalEncoding(self.hidden_dim)
        # FANformer 层
        self.layers = nn.ModuleList([
            FANformerLayer(self.hidden_dim, num_heads, norm_first, dropout)
            for _ in range(num_layers)
        ])

        self.out = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, input_ids):
        """
        src_texts: List[str]，输入是字符串列表
        """
        # (batch, seq_len)
        x = self.embedding(input_ids)

        x = x.transpose(0, 1) # (seq_len, batch, hidden)，适配 MultiheadAttention
        x = self.pos_encoder(x) # (seq_len, batch, hidden)

        # print(x)
        # causal mask: shape [seq_len, seq_len]
        seq_len = x.size(0)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask) # (seq_len, batch, hidden)

        # 输出到 (batch, seq_len, output_dim)
        logits = self.out(x).transpose(0, 1)  # [batch, seq_len, |VOCAB|]
        return logits

#################################


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, norm_first=True, dropout=0.1):
        super().__init__()
        self.norm_first = norm_first

        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # prenorm
        h = self.attn_norm(x) if self.norm_first else x
        # h = x

        # MultiheadAttention
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + self.dropout1(attn_out)

        # FFN
        h2 = self.ff_norm(x) if self.norm_first else x
        # h2 = x
        ff_out = self.ff(h2)
        x = x + self.dropout2(ff_out)
        return x

@register_model('Qwen2.5Embedding-Transformer')
class Qwen_Transformer(nn.Module):
    def __init__(self, pretrained_name="/home/gtxygyzb/models/Qwen/Qwen2.5-0.5B", output_dim=1,
                 num_layers=5, num_heads=12, norm_first=True, dropout=0.1, freeze_emb=True):
        super().__init__()
        # 加载预训练模型
        base_model = AutoModel.from_pretrained(pretrained_name)

        # 用预训练 embedding
        self.embedding = base_model.get_input_embeddings()
        self.hidden_dim = self.embedding.embedding_dim
        # 压缩和展开层
        
        print("hidden_dim:", self.hidden_dim)

        if freeze_emb:
            for p in self.embedding.parameters():
                p.requires_grad = False

        self.pos_encoder = RoPEPositionalEncoding(self.hidden_dim)
        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerLayer(self.hidden_dim, num_heads, norm_first, dropout)
            for _ in range(num_layers)
        ])

        self.out = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, input_ids):
        """
        src_texts: List[str]，输入是字符串列表
        """
        # (batch, seq_len)
        x = self.embedding(input_ids)

        x = x.transpose(0, 1) # (seq_len, batch, hidden)，适配 MultiheadAttention
        x = self.pos_encoder(x) # (seq_len, batch, hidden)

        seq_len = x.size(0)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask) # (seq_len, batch, hidden)

        # 输出到 (batch, seq_len, output_dim)
        logits = self.out(x).transpose(0, 1)  # [batch, seq_len, |VOCAB|]
        return logits
    
#########################

@register_model('OnlyNormNet_params')
class OnlyNormNet(nn.Module):
    def __init__(self, pretrained_name="/home/gtxygyzb/models/Qwen/Qwen2.5-0.5B", output_dim=1,
        num_layers=12, num_heads=12, norm_first=True, dropout=0.1, freeze_emb=True):
        super().__init__()
        # 加载预训练模型
        base_model = AutoModel.from_pretrained(pretrained_name)

        # 用预训练 embedding
        self.embedding = base_model.get_input_embeddings()
        self.hidden_dim = self.embedding.embedding_dim
        
        print("hidden_dim:", self.hidden_dim)

        if freeze_emb:
            for p in self.embedding.parameters():
                p.requires_grad = False
                self.pos_encoder = RoPEPositionalEncoding(self.hidden_dim)

        # Norm 层
        self.layers = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(num_layers)
        ])

        self.out = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        # (seq_len, batch, hidden)，适配 MultiheadAttention
        x = x.transpose(0, 1)

        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)

        # 输出到 (batch, seq_len, output_dim)
        logits = self.out(x).transpose(0, 1)  # [batch, seq_len, |VOCAB|]
        return logits

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )

        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


@register_model('KAN')
class KAN(nn.Module):
    def __init__(
        self,
        input_dim=1, 
        output_dim=1, 
        hidden_dim=128, 
        num_layers=3,
        grid_size=50,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        layers_hidden=[input_dim] + [hidden_dim] * num_layers + [output_dim]
        
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

########### Mamba Part ###########
# Mamba->MixerModel->Block->Mamba/GatedMLP
@register_model('Mamba')
class Mamba(nn.Module):
    def __init__(
        self,
        d_model=512,
        VOCAB_SIZE=200,
        num_layers=4,
        output_dim=1,
        num_heads=-1,
        initializer_cfg=None,
        freeze_emb=True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        
        self.num_layers = num_layers
        self.initializer_cfg = initializer_cfg or {}
        
        factory_kwargs = {"device": device, "dtype": dtype}

        base_model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B")
        
        pre_trained_embedding = base_model.get_input_embeddings()
        d_model = pre_trained_embedding.embedding_dim
        vocab_size = VOCAB_SIZE
        vocab_size = 100  
        
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=num_layers,
            d_intermediate=0,
            vocab_size=vocab_size,
            ssm_cfg={},
            attn_layer_idx={},
            attn_cfg={},
            rms_norm=True,
            initializer_cfg=initializer_cfg,
            fused_add_norm=True,
            residual_in_fp32=True,
            **factory_kwargs,
        )

        self.lm_head = nn.Linear(d_model, output_dim, bias=False, **factory_kwargs)
        self.apply(self._init_weights)
        self.backbone.embedding = pre_trained_embedding
        
        if freeze_emb:
            for p in self.backbone.embedding.parameters():
                p.requires_grad = False

    def _init_weights(self, module):

        n_layer = self.num_layers
        initializer_range = self.initializer_cfg.get("initializer_range", 0.02)
        rescale_prenorm_residual = self.initializer_cfg.get("rescale_prenorm_residual", True)
        n_residuals_per_layer = self.initializer_cfg.get("n_residuals_per_layer", 1)
        

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)
        

        if rescale_prenorm_residual:
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)

    def forward(self, input_ids, position_ids=None, inference_params=None, 
                num_last_tokens=0, **mixer_kwargs):

        hidden_states = self.backbone(
            input_ids, 
            inference_params=inference_params, 
            **mixer_kwargs
        )
        
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        
        lm_logits = self.lm_head(hidden_states)
        return lm_logits



# ------------------RWKV Part---------------------
CHUNK_LEN = 16
HEAD_SIZE = 64
flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
load(name="wind_backstepping", sources=[f'cuda/wkv7_cuda.cu', 'cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

@register_model('RWKV')
class RWKV(nn.Module):
    def __init__(self, d_model=512,
        head_size = 64,
        VOCAB_SIZE=200,
        num_layers=2,
        output_dim=1, freeze_emb=True, num_heads=-1):
        super().__init__()

    
        from argparse import Namespace
        args = Namespace()

        base_model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B")
        self.emb = base_model.get_input_embeddings()
        args.n_embd = self.emb.embedding_dim

        # args.n_embd = d_model
        args.vocab_size = VOCAB_SIZE
        args.n_layer = num_layers
        args.head_size = head_size
        assert args.n_embd % head_size == 0

        args.dim_att = args.n_embd
        args.dim_ffn = args.n_embd * 4
        self.args = args

        
        if freeze_emb:
            for p in self.emb.parameters():
                p.requires_grad = False
        # self.emb = nn.Embedding(200, args.n_embd,padding_idx=0)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, output_dim, bias=False)
        self.generate_init_weight(freeze_emb)

    
    def forward(self, idx):

        x = self.emb(idx)
        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)

        x = self.ln_out(x)
        x = self.head(x)

        return x
    
    def generate_init_weight(self, freeze_emb=True):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        n_params = 0
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            s3 = str(shape[3]) if len(shape) > 3 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}", end="")

            scale = 1.0
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n or n.endswith('_w') or n.endswith('_w1') or n.endswith('_w2') or n.endswith('_bias') or (".weight" not in n):
                if 'ln_x.weight' in n:
                    layer_scale = (1+int(n.split('.')[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p
                print()
            elif n == "emb.weight" and not freeze_emb:
                m[n] = p
                scale = -1e-4
                nn.init.uniform_(m[n], a=scale, b=-scale)
                print(f" [scale {scale}]")
            elif n == "head.weight":
                m[n] = p
                if self.args.vocab_size > self.args.n_embd:
                    scale = 0.5 * math.sqrt(self.args.vocab_size / self.args.n_embd)
                else:
                    scale = 0.5
                nn.init.orthogonal_(m[n], gain=scale)
                print(f" [scale {scale}]")
            else:
                assert n.endswith('.weight') # should always be true

                zero = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']

                for kk in zero:
                    if kk in n:
                        scale = 0

                for kk in [".att.key."]:
                    if kk in n:
                        scale = 0.1
                for kk in [".att.gate."]:
                    if kk in n:
                        scale = 0.1

                print(f" [scale {scale}]")
                m[n] = torch.empty((shape[0], shape[1]), device="cuda")

                # if self.args.accelerator.upper() == "GPU":
                #     m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                # else:
                #     m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=scale)

            m[n] = m[n].cpu()
            # m[n] = m[n].bfloat16()
            # if os.environ["RWKV_FLOAT_MODE"] == "fp16":
            #     m[n] = m[n].half()
            # elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
            #     m[n] = m[n].bfloat16()
            n_params += m[n].numel()

        print('model params', n_params)
        gc.collect()
        torch.cuda.empty_cache()
        return m
    
class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln0 = nn.LayerNorm(args.n_embd) # only used in block 0, should be fused with emb
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)
        

    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)

        xx, v_first = self.att(self.ln1(x), v_first)
        
        x = x + xx
        x = x + self.ffn(self.ln2(x))
        return x, v_first

class RWKV_Tmix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C-1) - 0.5
                zigzag[n] = ((n % N) - ((N-1) / 2)) / ((N-1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

            D_DECAY_LORA = max(32, int(round(  (2.5*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1,1,C) + 0.5 + zigzag*2.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = max(32, int(round(  (2.5*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C)-0.19 + zigzag*0.3 + linear*0.4)

            D_MV_LORA = max(32, int(round(  (1.7*(C**0.5))  /32)*32)) # suggestion
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+0.73 - linear*0.4)

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            D_GATE_LORA = max(32, int(round(  (5*(C**0.5))  /32)*32)) # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1,1,C)+0.71 - linear*0.1)
            self.k_a = nn.Parameter(torch.zeros(1,1,C)+1.02)
            self.r_k = nn.Parameter(torch.zeros(H,N)-0.04)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!

            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()

    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x = self.RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a).float()
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first
    
    def RUN_CUDA_RWKV7g(self, q, w, k, v, a, b):
        B,T,HC = q.shape
        q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
        q = q.to(torch.bfloat16)
        w = w.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        a = a.to(torch.bfloat16)
        b = b.to(torch.bfloat16)
        return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)
    

class RWKV_CMix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
        self.value.weight.data.zero_()

    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)




class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w,q,k,v,z,b):
        B,T,H,C = w.shape 
        assert T%CHUNK_LEN == 0 # if T%CHUNK_LEN != 0: pad your input to T%CHUNK_LEN == 0, or change CHUNK_LEN (will be slower)
        assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
        assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
        y = torch.empty_like(v)
        s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
        sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
        torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
        ctx.save_for_backward(w,q,k,v,z,b,s,sa)
        return y
    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype==torch.bfloat16 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        w,q,k,v,z,b,s,sa = ctx.saved_tensors
        dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
        torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
        return dw,dq,dk,dv,dz,db
    

