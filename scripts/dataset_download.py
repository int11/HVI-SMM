import kaggle
import gdown
import os
import zipfile


def gdrive_download(file_id, output_zip, extract_to):
    if os.path.exists(extract_to):
        print(f"[skip] {extract_to} already exists")
        return
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output=output_zip)
    with zipfile.ZipFile(output_zip, 'r') as zf:
        zf.extractall(extract_to)
    os.remove(output_zip)


# ── Kaggle datasets ─────────────────────────────────────────────────────────
kaggle.api.dataset_download_files("washingtongold/exdark-dataset", path="./datasets/ExDark", unzip=True)
kaggle.api.dataset_download_files("awsaf49/coco-2017-dataset", path="./datasets/coco", unzip=True)

# ── Google Drive datasets ────────────────────────────────────────────────────
os.makedirs("./datasets", exist_ok=True)

# LOL-v1
gdrive_download("1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H", "./datasets/LOL-v1.zip", "./datasets")

# LOL-v2
gdrive_download("1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U", "./datasets/LOL-v2.zip", "./datasets")

# SID
gdrive_download("1ECzih1CzWeBcBn8TQGWFGFi2YLmuRrrr", "./datasets/SID.zip", "./datasets")

# MIT-Adobe FiveK
gdrive_download("11HEUmchFXyepI4v3dhjnDnmhW_DgwfRR", "./datasets/FiveK.zip", "./datasets")

# NTIRE 2024
gdrive_download("1Js9yHmV0xAWhT5oJKzfx6oOr_7k5hcNg", "./datasets/NTIRE2024_train_input.zip", "./datasets")
gdrive_download("1PUJgJiEyrIj5TgwcQlFvVGuIe3_PXMLY", "./datasets/NTIRE2024_train_GT.zip", "./datasets")
gdrive_download("1Z9XJ1x2Ibh_WUo59l_AI9gQAbyhpIswe", "./datasets/NTIRE2024_minival.zip", "./datasets")
