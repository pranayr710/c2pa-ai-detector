import hashlib
import os
from PIL import Image, PngImagePlugin
import piexif


# ==========================================================
# Pixel Hash (Integrity)
# ==========================================================
def get_pixel_hash(image_path):
    """
    Generate SHA-256 hash of image pixel data
    (used for integrity comparison, not provenance)
    """
    try:
        img = Image.open(image_path).convert("RGB")
        return hashlib.sha256(img.tobytes()).hexdigest()
    except Exception as e:
        raise ValueError(f"Error reading image pixels: {str(e)}")


# ==========================================================
# File Hash (Stable for JPG)
# ==========================================================
def get_file_hash(image_path):
    """
    Generate SHA-256 hash of full file bytes
    (used for JPEG sealing to avoid false tampering)
    """
    try:
        with open(image_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        raise ValueError(f"Error reading image file: {str(e)}")


# ==========================================================
# Embed Lightweight Seal
# ==========================================================
def embed_seal(input_image_path, output_image_path):
    """
    Embed a lightweight cryptographic seal into an image.

    - PNG  : pixel hash stored as metadata
    - JPG  : file hash stored in EXIF UserComment

    Returns:
        SHA-256 hash string
    """
    try:
        img = Image.open(input_image_path).convert("RGB")
        ext = os.path.splitext(output_image_path)[1].lower()

        # ---------- PNG ----------
        if ext == ".png":
            image_hash = get_pixel_hash(input_image_path)
            meta = PngImagePlugin.PngInfo()
            meta.add_text("c2pa_seal", image_hash)
            img.save(output_image_path, format="PNG", pnginfo=meta)

        # ---------- JPG ----------
        elif ext in [".jpg", ".jpeg"]:
            img.save(output_image_path)   # save first
            image_hash = get_file_hash(output_image_path)

            exif_dict = {"Exif": {}}
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = image_hash.encode("utf-8")
            exif_bytes = piexif.dump(exif_dict)
            img.save(output_image_path, exif=exif_bytes)

        else:
            raise ValueError("Unsupported image format")

        return image_hash

    except Exception as e:
        raise ValueError(f"Failed to embed seal: {str(e)}")


# ==========================================================
# Detect Lightweight Seal
# ==========================================================
def get_c2pa_seal(image_path):
    """
    Detect lightweight seal embedded by embed_seal().
    """
    try:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        img = Image.open(image_path)

        # ---------- PNG ----------
        if image_path.lower().endswith(".png"):
            return img.info.get("c2pa_seal", None)

        # ---------- JPG ----------
        if image_path.lower().endswith((".jpg", ".jpeg")):
            exif_bytes = img.info.get("exif", b"")
            if not exif_bytes:
                return None

            exif = piexif.load(exif_bytes)
            comment = exif["Exif"].get(piexif.ExifIFD.UserComment)
            return comment.decode("utf-8") if comment else None

        return None

    except Exception as e:
        raise ValueError(f"Error reading seal metadata: {str(e)}")
