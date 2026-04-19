import kaggle

kaggle.api.dataset_download_files("washingtongold/exdark-dataset", path="./datasets/ExDark", unzip=True)
kaggle.api.dataset_download_files("weipengzhang/adobe-fivek",       path="./datasets/FiveK",  unzip=True)
kaggle.api.dataset_download_files("awsaf49/coco-2017-dataset",      path="./datasets/coco",   unzip=True)
