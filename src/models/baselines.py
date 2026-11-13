import torch
import torch.nn as nn

# ==========================================
# 1. TransRR (Zhao et al., 2023)
# ==========================================
class KernelInceptionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        b = d_model // 4
        self.k3 = nn.Conv1d(d_model, b, kernel_size=3, padding=1)
        self.k5 = nn.Conv1d(d_model, b, kernel_size=5, padding=2)
        self.k7 = nn.Conv1d(d_model, b, kernel_size=7, padding=3)
        self.k9 = nn.Conv1d(d_model, b, kernel_size=9, padding=4)

    def forward(self, x):
        return torch.cat([self.k3(x), self.k5(x), self.k7(x), self.k9(x)], dim=1)

class DilationInceptionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        b = d_model // 4
        self.d1 = nn.Conv1d(d_model, b, kernel_size=5, padding=2, dilation=1)
        self.d4 = nn.Conv1d(d_model, b, kernel_size=5, padding=8, dilation=4)
        self.d8 = nn.Conv1d(d_model, b, kernel_size=5, padding=16, dilation=8)
        self.d16 = nn.Conv1d(d_model, b, kernel_size=5, padding=32, dilation=16)

    def forward(self, x):
        return torch.cat([self.d1(x), self.d4(x), self.d8(x), self.d16(x)], dim=1)

class TransRRModule(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.kernel_inc = KernelInceptionBlock(d_model)
        self.dilation_inc = DilationInceptionBlock(d_model)
        self.dropout = nn.Dropout(0.1)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_out)
        
        x_c = x.permute(0, 2, 1)
        x_k = self.kernel_inc(x_c)
        x_d = self.dilation_inc(x_k)
        x_conv = self.conv1d(self.dropout(x_d))
        
        x_out = x_conv.permute(0, 2, 1)
        return self.norm2(x + x_out)

class TransRR(nn.Module):
    def __init__(self, in_channels=1, d_model=256, n_heads=8, num_layers=4):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, d_model, kernel_size=7, stride=2, padding=3)
        self.layers = nn.ModuleList([TransRRModule(d_model, n_heads) for _ in range(num_layers)])
        self.dnn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 256), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(256, 64), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(64, 16), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.input_proj(x).permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)
        x = x.permute(0, 2, 1)
        return self.dnn(x)


# ==========================================
# 2. Hybrid CNN-LSTM (Chin et al., 2024)
# ==========================================
class HybridCNNLSTM(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.MaxPool1d(2),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        feat = self.conv(x).permute(0, 2, 1)
        out, _ = self.lstm(feat)
        return self.fc(out[:, -1, :]) 


# ==========================================
# 3. ConvMixer (Chowdhury et al., 2022)
# ==========================================
class ConvMixerBlock1D(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=dim, padding="same"),
            nn.GELU(),
            nn.BatchNorm1d(dim)
        )
        self.pw = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.GELU(),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        return self.pw(x + self.res(x))

class ConvMixer(nn.Module):
    def __init__(self, in_channels=1, dim=256, depth=8, patch_size=10, kernel_size=7):
        super().__init__()
        self.pe = nn.Sequential(
            nn.Conv1d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm1d(dim)
        )
        self.blocks = nn.ModuleList([ConvMixerBlock1D(dim, kernel_size) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(dim, 1)
        )

    def forward(self, x):
        x = self.pe(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


# ==========================================
# 4. Resptrack 1D-CNN (Jash et al., 2026)
# ==========================================
class Resptrack1DCNN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 96, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(96, 112, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(112, 68, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(68, 12, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool1d(2)
        )
        
        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool1d(30), 
            nn.Flatten(),
            nn.Linear(12 * 30, 400), nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(400, 32), nn.ReLU(),
            nn.Linear(32, 208), nn.ReLU(),
            nn.Linear(208, 1)
        )

    def forward(self, x):
        return self.dense(self.conv(x))