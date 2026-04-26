from pathlib import Path
from PIL import Image

_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}


def is_image_file(filename):
    return Path(filename).suffix.lower() in _IMAGE_EXTS


def list_images(directory, recursive=False):
    """List sorted absolute image paths under `directory`."""
    p = Path(directory)
    if not p.exists():
        return []
    if recursive:
        files = [str(f) for f in p.rglob('*') if f.is_file() and is_image_file(f.name)]
    else:
        files = [str(f) for f in p.iterdir() if f.is_file() and is_image_file(f.name)]
    files.sort()
    return files


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img
