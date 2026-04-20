import kaggle
import gdown
import os
import shutil
import zipfile


def gdrive_download(file_id, output_zip, extract_to):
    if os.path.exists(extract_to) and os.listdir(extract_to):
        print(f"[skip] {extract_to} already exists")
        return
    os.makedirs(extract_to, exist_ok=True)
    try:
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output=output_zip, fuzzy=True)
        with zipfile.ZipFile(output_zip, 'r') as zf:
            members = zf.namelist()
            top = os.path.commonprefix(members).rstrip("/") + "/"
            strip = top if all(m.startswith(top) for m in members) else ""
            for member in members:
                target = member[len(strip):] if strip else member
                if not target:
                    continue
                dest = os.path.join(extract_to, target)
                if member.endswith("/"):
                    os.makedirs(dest, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
        os.remove(output_zip)
    except Exception as e:
        print(f"[error] {extract_to}: {e}")
        shutil.rmtree(extract_to, ignore_errors=True)
        if os.path.exists(output_zip):
            os.remove(output_zip)


# ── Kaggle datasets ─────────────────────────────────────────────────────────
def kaggle_download(dataset, path):
    if os.path.exists(path) and os.listdir(path):
        print(f"[skip] {path} already exists")
        return
    os.makedirs(path, exist_ok=True)
    kaggle.api.dataset_download_files(dataset, path=path, unzip=True)

kaggle_download("washingtongold/exdark-dataset", "./datasets/ExDark")
kaggle_download("awsaf49/coco-2017-dataset", "./datasets/coco")

# ── Google Drive datasets ────────────────────────────────────────────────────
# LOL-v1
gdrive_download("1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H", "./datasets/LOL-v1.zip", "./datasets/LOL-v1")

# LOL-v2
gdrive_download("1Ou9EljYZW8o5dbDCf9R34FS8Pd8kEp2U", "./datasets/LOL-v2.zip", "./datasets/LOL-v2")

# SID
gdrive_download("1ECzih1CzWeBcBn8TQGWFGFi2YLmuRrrr", "./datasets/SID.zip", "./datasets/SID")

# MIT-Adobe FiveK
gdrive_download("11HEUmchFXyepI4v3dhjnDnmhW_DgwfRR", "./datasets/FiveK.zip", "./datasets/FiveK")

# NTIRE 2024
gdrive_download("1Js9yHmV0xAWhT5oJKzfx6oOr_7k5hcNg", "./datasets/NTIRE2024_train_input.zip", "./datasets/NTIRE2024_train_input")
gdrive_download("1PUJgJiEyrIj5TgwcQlFvVGuIe3_PXMLY", "./datasets/NTIRE2024_train_GT.zip", "./datasets/NTIRE2024_train_GT")
gdrive_download("1Z9XJ1x2Ibh_WUo59l_AI9gQAbyhpIswe", "./datasets/NTIRE2024_minival.zip", "./datasets/NTIRE2024_minival")
