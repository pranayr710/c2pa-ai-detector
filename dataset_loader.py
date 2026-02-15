import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    """
    Image Dataset with Train / Validation Split

    Label Convention:
    0 = REAL
    1 = AI
    """

    def __init__(self, root_dir, split="train", split_ratio=0.8):
        self.data = []

        # ImageNet normalization (CRITICAL for EfficientNet)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        real_dir = os.path.join(root_dir, "real")
        ai_dir = os.path.join(root_dir, "ai")

        real_images = [
            os.path.join(real_dir, f)
            for f in os.listdir(real_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        ai_images = [
            os.path.join(ai_dir, f)
            for f in os.listdir(ai_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        random.shuffle(real_images)
        random.shuffle(ai_images)

        real_split = int(len(real_images) * split_ratio)
        ai_split = int(len(ai_images) * split_ratio)

        if split == "train":
            self.data = (
                [(p, 0) for p in real_images[:real_split]] +
                [(p, 1) for p in ai_images[:ai_split]]
            )
        else:
            self.data = (
                [(p, 0) for p in real_images[real_split:]] +
                [(p, 1) for p in ai_images[ai_split:]]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label
