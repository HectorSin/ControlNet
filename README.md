# ControlNet을 활용한 양자화 프로젝트

## 환경 설정
```
git clone https://github.com/HectorSin/ControlNet con11

cd con/models
# cd ./models
# 1.45 GB
curl -LO https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_mlsd.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_seg.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_inpaint.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15s2_lineart_anime.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11e_sd15_shuffle.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11e_sd15_ip2p.pth
curl -LO https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth
curl -LO https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt
```

베이스 도커 이미지
```
nvidia/cuda:11.1.1-devel-ubuntu20.04
```

콘다 환경
```
# 1. Anaconda 설치 스크립트 다운로드
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh

# 2. 설치 스크립트 실행
bash Anaconda3-2023.03-Linux-x86_64.sh

# 3. 설치 과정에 따라 진행 (라이센스 동의, 설치 경로 선택 등)
# 설치가 끝나면 Anaconda를 활성화하기 위해 다음 명령을 실행합니다.
source ~/.bashrc
```

가상환경 설치및 필수 패키지 설치
```
conda env create -f environment.yaml

conda activate control-v11

sudo apt-get update
sudo apt-get install libsm6 libxext6 libxrender-dev
```

# 프로젝트 설명


## 성능지표

## 결과물

## 양자화