import torch.nn as nn

class GANClassifier(nn.Module):
    """
    Multi-class CNN for identifying which AI generator created an image.
    Analyzes both spatial and frequency domain patterns.
    
    Input: 3-channel image (3 x 224 x 224)
    Output: N classes (real + each generator type)
    """
    def __init__(self, num_classes=5):
        super().__init__()

        # Spatial feature extractor
        self.spatial = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # 256 x 4 x 4
        )

        # Frequency feature extractor (operates on grayscale FFT magnitude)
        self.freq = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # 128 x 4 x 4
        )

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4 + 128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        import torch
        import torch.fft as fft

        # Spatial stream
        spatial_feat = self.spatial(x)

        # Frequency stream: grayscale -> FFT -> magnitude
        gray = torch.mean(x, dim=1, keepdim=True)
        freq_data = fft.fft2(gray)
        freq_mag = torch.abs(freq_data)
        freq_feat = self.freq(freq_mag)

        # Flatten and combine
        spatial_feat = spatial_feat.view(spatial_feat.size(0), -1)
        freq_feat = freq_feat.view(freq_feat.size(0), -1)
        combined = torch.cat((spatial_feat, freq_feat), dim=1)

        return self.classifier(combined)
