import json
from flask import Flask, request, render_template, send_file
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageFilter
import torch
import os

app = Flask(__name__)

# 설정 파일 로드
with open("settings.json", "r") as f:
    settings = json.load(f)

# 모델 설정 및 초기화
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "stabilityai/stable-diffusion-2-inpainting"
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id).to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # 이미지 업로드 및 원본 크기 추출
    image_file = request.files['image']
    input_image = Image.open(image_file).convert("RGB")
    width, height = input_image.size  # 원본 크기

    # width와 height를 8의 배수로 맞춤
    new_width = (width // 8) * 8
    new_height = (height // 8) * 8
    input_image = input_image.resize((new_width, new_height), Image.LANCZOS)

    # 마스크 생성: 2x3 그리드 중 중앙(11) 위치에만 마스킹
    mask = Image.new("L", (new_width, new_height), 0)  # 검은색 기본 마스크
    grid_width, grid_height = new_width // 3, new_height // 2
    mask_x1, mask_y1 = grid_width, grid_height
    mask_x2, mask_y2 = grid_width * 2, grid_height * 2
    mask.paste(255, (mask_x1, mask_y1, mask_x2, mask_y2))  # 중앙 영역 흰색 마스킹

    # 마스크 블러 처리
    if settings.get("mask_blur", 0) > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=settings["mask_blur"]))

    # inpainting 파라미터 설정
    inpaint_params = {
        "prompt": settings["prompt"],
        "image": input_image,
        "mask_image": mask,
        "num_inference_steps": settings["sampling_steps"],
        "guidance_scale": settings["cfg_scale"],
        "width": new_width,
        "height": new_height,
        "negative_prompt": settings["negative_prompt"],
        "strength": settings["denoising_strength"]
    }

    # inpainting 수행
    with torch.no_grad():
        result = pipe(**inpaint_params).images[0]

    # 결과 이미지 저장
    output_path = "static/output.png"
    result.save(output_path)

    return send_file(output_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
