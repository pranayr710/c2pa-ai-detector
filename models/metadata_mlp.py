import torch.nn as nn

class MetadataMLP(nn.Module):
    """
    Multi-Layer Perceptron for metadata-based authenticity scoring.
    
    Input: Feature vector of 15 metadata features
    Output: 2 classes (Real=0, AI=1)
    
    Features extracted from images:
    1.  has_camera_make       (0 or 1)
    2.  has_camera_model      (0 or 1)
    3.  has_gps               (0 or 1)
    4.  has_datetime           (0 or 1)
    5.  has_datetime_original  (0 or 1)
    6.  has_software           (0 or 1)
    7.  has_ai_software        (0 or 1) - known AI tool signatures
    8.  has_ai_metadata_key    (0 or 1) - keys like 'prompt', 'seed'
    9.  exif_tag_count         (normalized 0-1)
    10. file_size_kb           (normalized 0-1)
    11. is_square              (0 or 1) - AI images are often square
    12. is_power_of_2          (0 or 1) - 512x512, 1024x1024 etc.
    13. width_normalized       (normalized 0-1)
    14. height_normalized      (normalized 0-1)
    15. has_lens_info          (0 or 1)
    """
    def __init__(self, input_size=15):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)
