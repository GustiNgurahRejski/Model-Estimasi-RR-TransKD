import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class PPGPatchTransformer(nn.Module):
    def __init__(self, in_channels=1, patch_size=25, d_model=128, n_heads=4, num_layers=4, dim_feedforward=256, dropout=0.1, is_student=False):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model

        # 1. Feature Extractor Frontend
        self.frontend = nn.Sequential(
            nn.Conv1d(in_channels, d_model // 2, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=patch_size, stride=patch_size // 2),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward if dim_feedforward else d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Regression Head
        self.fc_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        feat_maps = self.frontend(x)
        patches = feat_maps.permute(0, 2, 1)
        patches = self.pos_encoder(patches)

        feat_seq = self.transformer(patches)
        feat_global = feat_seq.permute(0, 2, 1)
        out = self.fc_head(feat_global)

        return out, feat_seq, feat_global

def build_transformer(cfg_obj=None, is_student=False, **kwargs):
    if cfg_obj is not None:
        d_model = getattr(cfg_obj, "d_model", 32 if is_student else 128)
        n_heads = getattr(cfg_obj, "n_heads", getattr(cfg_obj, "nhead", 2 if is_student else 8))
        num_layers = getattr(cfg_obj, "num_layers", 2 if is_student else 4)
        dim_ff = getattr(cfg_obj, "dim_feedforward", 64 if is_student else 256)
        patch_size = getattr(cfg_obj, "patch_size", 25)
        dropout = getattr(cfg_obj, "dropout", 0.1)
    else:
        d_model = kwargs.get("d_model", 32 if is_student else 128)
        n_heads = kwargs.get("n_heads", 2 if is_student else 8)
        num_layers = kwargs.get("num_layers", 2 if is_student else 4)
        dim_ff = kwargs.get("dim_feedforward", 64 if is_student else 256)
        patch_size = kwargs.get("patch_size", 25)
        dropout = kwargs.get("dropout", 0.1)

    return PPGPatchTransformer(
        in_channels=1,
        patch_size=patch_size,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_feedforward=dim_ff,
        dropout=dropout,
        is_student=is_student
    )