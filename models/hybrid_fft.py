import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

class HybridFFT(nn.Module):
    def __init__(self):
        super().__init__()

        self.spatial = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

        self.freq = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

        self.fc = nn.Linear(128 * 28 * 28 + 64 * 56 * 56, 2)

    def forward(self, x):
        spatial_feat = self.spatial(x)

        gray = torch.mean(x, dim=1, keepdim=True)
        freq_feat = fft.fft2(gray)
        freq_feat = torch.abs(freq_feat)
        freq_feat = self.freq(freq_feat)

        spatial_feat = spatial_feat.view(spatial_feat.size(0), -1)
        freq_feat = freq_feat.view(freq_feat.size(0), -1)

        combined = torch.cat((spatial_feat, freq_feat), dim=1)
        return self.fc(combined)
