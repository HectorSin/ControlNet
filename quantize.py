import torch
import torch.quantization as quantization
from cldm.model import create_model, load_state_dict  # cldm.model에서 필요한 함수 임포트
import os

# 1. 모델 정의 및 로드
model_name = 'control_v11p_sd15_scribble'
yaml_path = f'models/{model_name}.yaml'  # YAML 파일 경로
state_dict_path = f'models/{model_name}.pth'  # 상태 딕셔너리 파일 경로

# 모델 생성 및 상태 딕셔너리 로드
print("모델 생성 중...")
model = create_model(yaml_path).cpu()  # YAML 파일을 사용하여 모델 생성
print("상태 딕셔너리 로드 중...")
state_dict = load_state_dict(state_dict_path, location='cpu')  # 상태 딕셔너리 로드

# 모델에 상태 딕셔너리 로드
model.load_state_dict(state_dict, strict=False)  # strict=False를 사용하여 일부 키가 누락된 경우 무시
model.eval()  # 양자화는 추론 모드에서 수행됨

# 2. 선택한 백엔드 및 기본 qconfig 설정
backend = "qnnpack"  # 또는 "x86", "fbgemm" 등 지원하는 백엔드 중 선택
torch.backends.quantized.engine = backend
qconfig = torch.quantization.get_default_qconfig(backend)
model.qconfig = qconfig
print("백엔드 및 qconfig 설정 완료...")

# 3. 임베딩 레이어에 적절한 qconfig 적용
# 여기서 'embedding_layer_name'을 실제 모델의 임베딩 레이어 이름으로 대체해야 합니다
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Embedding):
        module.qconfig = quantization.float_qparams_weight_only_qconfig
        print(f"{name}에 대해 float_qparams_weight_only_qconfig가 설정되었습니다.")

# 4. 모델 준비 및 양자화 관측 수행
print("모델 양자화 준비 중...")
torch.quantization.prepare(model, inplace=True)
print("양자화 준비 완료...")

# 5. 모델에 양자화 적용
print("양자화 변환 중...")
torch.quantization.convert(model, inplace=True)
print("양자화 변환 완료...")

# 6. 양자화된 모델 저장
save_dir = 'q_models'
os.makedirs(save_dir, exist_ok=True)  # 디렉토리 생성
save_path = os.path.join(save_dir, f'{model_name}_quantized.pth')
torch.save(model.state_dict(), save_path)
print(f"양자화된 모델이 {save_path}에 저장되었습니다.")
