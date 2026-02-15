import os
from utils import embed_seal

FOLDER = r"C:\Users\Vasala Vignesh\Documents\pranay\image_verifier\image_verifier\dataset_sealed"

for name in sorted(os.listdir(FOLDER)):
    if name.lower().endswith((".jpg", ".jpeg")):
        path = os.path.join(FOLDER, name)
        embed_seal(path, path)   # overwrite with EXIF seal
        print("Resealed:", name)

print("✅ All JPGs resealed with EXIF")
