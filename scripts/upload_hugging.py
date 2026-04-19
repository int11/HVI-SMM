import torch
from safetensors.torch import save_file
from huggingface_hub import login, upload_file
import tempfile
import os

# Hugging Face 로그인
login()

# Step 1: checkpoint 로드
checkpoint = torch.load("./weights/lolv2_syn/lolv2_syn.pth")

# Step 2: state_dict 추출
model_state_dict = checkpoint['model_state_dict']

# Step 3: 임시 파일에 safetensors 형식으로 저장
with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as tmp:
    tmp_path = tmp.name

save_file(model_state_dict, tmp_path)

# Step 4: Hub에 업로드
upload_file(
    path_or_fileobj=tmp_path,
    path_in_repo='model.safetensors',
    repo_id='akyaa/HVI-SMM-LOLv2-syn',
    repo_type='model'
)

# Step 5: 임시 파일 삭제
os.remove(tmp_path)

print("✅ safetensors 형식으로 성공적으로 업로드되었습니다!")