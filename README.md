# ControlNet 양자화 및 성능 최적화 프로젝트

ControlNet은 간단한 스케치를 이미지로 변환하는 강력한 모델입니다. 이 프로젝트에서는 ControlNet 모델을 양자화하고 성능을 최적화하여 추론 시간을 단축하는 것을 목표로 합니다. 또한, 이를 갈틱폰 게임에 활용하여 새로운 모드를 구현하는 아이디어를 테스트합니다.

## 프로젝트 개요

이 프로젝트는 다음과 같은 목표를 가지고 있습니다:

1. **ControlNet 모델의 양자화 및 프루닝**: 모델의 크기를 줄이고 추론 속도를 높이기 위한 작업.
2. **성능 지표 평가**: FID, PSNR, SSIM 등 다양한 성능 지표를 활용하여 모델의 성능을 평가.
3. **Gradio 인터페이스 개선**: Gradio를 사용하여 모델의 결과물을 직관적으로 시각화하고, 사용자 인터페이스를 개선.
4. **ControlNet 1.0과 1.1 비교**: 두 버전 간의 성능 차이를 비교하고, 논문 리뷰를 통해 변경 사항을 확인.
5. **갈틱폰 게임에 AI Bot 도입**: ControlNet을 활용하여 AI가 참여하는 갈틱폰 게임 모드를 구현.

## 주요 기능

- ControlNet 모델의 양자화를 통해 추론 시간 단축
- FID, PSNR, SSIM 등의 성능 지표를 활용한 모델 성능 평가
- Gradio를 통한 인터페이스 제공 및 사용자 인터랙션 개선
- ControlNet 1.0과 1.1 버전 간의 성능 비교

## 설치 방법

이 프로젝트를 로컬에서 실행하기 위해 다음 단계를 따르세요:

1. 이 저장소를 클론합니다:

    ```bash
    git clone https://github.com/HectorSin/ControlNet.git
    cd ControlNet
    ```

2. Conda 가상환경을 설정하고 필요한 패키지를 설치합니다:

    ```bash
    conda env create -f environment.yaml
    conda activate control-v11
    sudo apt-get update
    sudo apt-get install libsm6 libxext6 libxrender-dev
    ```

3. 필요한 모델 파일을 다운로드하여 `models/` 디렉토리에 저장합니다:

```
cd ./models

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

## 사용 방법

ControlNet 모델을 실행하여 이미지를 생성하고, 성능을 평가하는 방법:

1. `test_imgs` 디렉토리에 입력 이미지를 추가합니다.
2. ControlNet 모델을 실행하여 이미지를 생성합니다:

    ```bash
    python gradio_app.py  # Gradio 인터페이스를 통해 사용 가능
    ```

3. 생성된 이미지의 성능을 평가하고 시각화하기 위해 다음 명령을 실행합니다:

    ```bash
    python visualize.py  # 성능 지표를 시각화하여 결과물로 저장
    ```

## 성능지표

### FID (Frechet Inception Distance)
FID는 생성된 이미지와 실제 이미지의 통계적 유사성을 측정하는 지표입니다. 낮을수록 품질이 높습니다.

### PSNR (Peak Signal-to-Noise Ratio)
PSNR은 두 이미지 간의 최대 신호 대 잡음 비율을 평가합니다. 값이 클수록 두 이미지가 유사함을 나타냅니다.

### SSIM (Structural Similarity Index)
SSIM은 두 이미지 간의 구조적 유사성을 평가합니다. 값이 1에 가까울수록 유사성이 높습니다.

## 테스트 환경 및 성능 평가

- **테스트 환경**: RTX3090 GPU, Python 3.8, PyTorch 1.12.1
- **평가 지표**: FID, PSNR, SSIM
- **모델 성능 평가**: 양자화 전후의 모델 성능 및 추론 시간 비교

## 역할 및 일정

### 역할 분담

- **모델 구현 및 환경 체크**: 신재현
- **프로젝트 구현 및 테스트**: 현동철
- **모델 양자화 및 경량화 작업**: 최호재
- **문헌 조사 및 발표 자료 준비**: 팀원 전체

## 연락처

추가적인 정보나 질문이 있으면 [이메일 주소](mailto:kkang15634@ajou.ac.kr)로 연락해 주세요.