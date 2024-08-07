import os
import matplotlib.pyplot as plt
from PIL import Image
from fid import compute_fid
from psnr import compute_psnr
from ssim import compute_ssim
import config

# 경로 설정
input_dir = "test_imgs"
if config.status == "Pure":
    output_dir = "out_imgs"
else:
    output_dir = "q_out_imgs"
    
output_file = f"visualization_{config.status}.png"  # 저장할 이미지 파일 이름

# 이미지 파일 목록 가져오기
input_images = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))])

# 시각화하기 위한 설정
fig, axes = plt.subplots(len(input_images), 3, figsize=(15, 5 * len(input_images)))

for i, input_image_name in enumerate(input_images):
    # 입력 이미지 불러오기
    input_image_path = os.path.join(input_dir, input_image_name)
    input_image = Image.open(input_image_path)

    # 출력 이미지 1 불러오기
    output_image_path_0 = os.path.join(output_dir, f"{os.path.splitext(input_image_name)[0]}_output_0.png")
    output_image_0 = Image.open(output_image_path_0)

    # 출력 이미지 2 불러오기
    output_image_path_1 = os.path.join(output_dir, f"{os.path.splitext(input_image_name)[0]}_output_1.png")
    output_image_1 = Image.open(output_image_path_1)

    # 각 출력 이미지에 대해 FID, PSNR 및 SSIM 계산
    fid_score_0 = compute_fid(input_image_path, output_image_path_0)
    fid_score_1 = compute_fid(input_image_path, output_image_path_1)
    psnr_score_0 = compute_psnr(input_image_path, output_image_path_0)
    psnr_score_1 = compute_psnr(input_image_path, output_image_path_1)
    ssim_score_0 = compute_ssim(input_image_path, output_image_path_0)
    ssim_score_1 = compute_ssim(input_image_path, output_image_path_1)

    # 입력 이미지 시각화
    axes[i, 0].imshow(input_image)
    axes[i, 0].set_title(f"Input Image {i+1}")
    axes[i, 0].axis("off")

    # 출력 이미지 1 시각화 (FID, PSNR, SSIM 포함)
    axes[i, 1].imshow(output_image_0)
    axes[i, 1].set_title(f"Output Image {i+1} - 1\nFID: {fid_score_0:.2f}, PSNR: {psnr_score_0:.2f}, SSIM: {ssim_score_0:.2f}")
    axes[i, 1].axis("off")

    # 출력 이미지 2 시각화 (FID, PSNR, SSIM 포함)
    axes[i, 2].imshow(output_image_1)
    axes[i, 2].set_title(f"Output Image {i+1} - 2\nFID: {fid_score_1:.2f}, PSNR: {psnr_score_1:.2f}, SSIM: {ssim_score_1:.2f}")
    axes[i, 2].axis("off")

# 그래프 사이 간격 조정
plt.tight_layout()

# 결과물을 이미지 파일로 저장
plt.savefig(output_file)

# plt.show() 생략, 대신 이미지로 저장
print(f"Visualization saved as {output_file}")