import hashlib
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def extract_seal(image_path):
    """
    Extract custom seal metadata from image (if exists).
    Returns stored hash or None.
    """
    try:
        img = Image.open(image_path)
        meta = img.info
        return meta.get("seal_hash")
    except Exception:
        return None


def compute_image_hash(image_path):
    """
    Compute SHA-256 hash of image pixels.
    """
    img = Image.open(image_path).convert("RGB")
    return hashlib.sha256(img.tobytes()).hexdigest()
