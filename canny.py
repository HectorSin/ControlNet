import os
import cv2
import einops
import numpy as np
import torch
import random
from PIL import Image
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import config

# 모델 및 샘플러 설정
preprocessor = None
model_name = 'control_v11p_sd15_canny'
model = create_model(f'./models/{model_name}.yaml').cpu()

# 양자화된 모델 로드
state_dict = load_state_dict(f'./q_models/{model_name}.pth', location='cuda')
for key in state_dict.keys():
    if "weight" in key or "bias" in key:
        state_dict[key] = state_dict[key].dequantize()

model.load_state_dict(state_dict, strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

# 처리 함수 정의
def process_image(det, input_image_path, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    global preprocessor

    if det == 'Canny':
        if not isinstance(preprocessor, CannyDetector):
            preprocessor = CannyDetector()

    with torch.no_grad():
        # 이미지 로드 및 전처리
        input_image = np.array(Image.open(input_image_path).convert("RGB"))
        input_image = HWC3(input_image)

        if det == 'None':
            detected_map = input_image.copy()
        else:
            detected_map = preprocessor(resize_image(input_image, detect_resolution), low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results

# 경로 설정
input_dir = "test_imgs"
output_dir = "q_out_imgs"
os.makedirs(output_dir, exist_ok=True)

# 파라미터 설정
det = 'Canny'
a_prompt = 'best quality'
n_prompt = 'lowres, bad anatomy, bad hands, cropped, worst quality'
num_samples = 1
image_resolution = 512
detect_resolution = 512
ddim_steps = 20
guess_mode = False
strength = 1.0
scale = 9.0
seed = 12345
eta = 0.0
low_threshold = 100
high_threshold = 200

# 이미지 처리 및 저장
for image_file in os.listdir(input_dir):
    input_image_path = os.path.join(input_dir, image_file)
    prompt = os.path.splitext(image_file)[0]
    
    results = process_image(det, input_image_path, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold)
    
    for i, result in enumerate(results):
        output_image_path = os.path.join(output_dir, f"{prompt}_output_{i}.png")
        Image.fromarray(result).save(output_image_path)

    print(f"Processed and saved {image_file} as {output_image_path}")
