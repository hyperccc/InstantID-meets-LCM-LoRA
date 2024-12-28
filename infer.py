import cv2
import torch
import numpy as np
from PIL import Image

from diffusers.utils import load_image
from diffusers.models import ControlNetModel

from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps
from diffusers import LCMScheduler
import os
from datetime import datetime


def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


if __name__ == "__main__":

    # Load face encoder
    app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Path to InstantID models
    face_adapter = f'./checkpoints/ip-adapter.bin'
    controlnet_path = f'./checkpoints/ControlNetModel'

    # Load pipeline
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

    base_model_path = 'stabilityai/stable-diffusion-xl-base-1.0'
    lcm_lora_path = "./checkpoints/pytorch_lora_weights.safetensors"

    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.cuda()
    pipe.load_ip_adapter_instantid(face_adapter)
    # 在加载 LCM_LoRA 之前保存原始调度器
    original_scheduler = pipe.scheduler
    original_scheduler_config = pipe.scheduler.config
    #LCM_LoRA
    pipe.load_lora_weights(lcm_lora_path)
    pipe.fuse_lora()
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # num_inference_steps = 10
    # guidance_scale = 0

    # Infer setting
    # prompt = "analog film photo of a man. faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, masterpiece, best quality"
    # n_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured"
    # prompt= "cinematic film still . shallow depth of field, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy"
    n_prompt= "cross-eyed, anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, watermark, painting, drawing, illustration, glitch,deformed, mutated"
    # prompt="a close-up photography of a man standing in the rain at night, in a street lit by lamps, leica 35mm summilux" #close-up  standing in the rain at night
    prompt = "A photographic image close-up of a beautiful woman in a lightning storm, standing in a moonlit forest at midnight."
    face_image = load_image("./examples/Trump.png")  #examples/luo.jpg examples/cheng.jpg
    face_image = resize_img(face_image)

    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
    face_emb = face_info['embedding']
    face_kps = draw_kps(face_image, face_info['kps'])
    output_dir = "/data4/students/user24215459/InstantID-main/results/infer"
    os.makedirs(output_dir, exist_ok=True)
    #unload LoRA weights
    # pipe.unload_lora_weights()
    # pipe.scheduler = type(original_scheduler).from_config(original_scheduler_config)

    images = []
    for steps in range(9):
        print(f"Step {steps+1}: Starting generation...")
        try:
            image = pipe(
                prompt=prompt,
                negative_prompt=n_prompt,
                image_embeds=face_emb,
                image=face_kps,
                controlnet_conditioning_scale=0.8,
                ip_adapter_scale=0.8,
                num_inference_steps=steps + 1,  #30
                guidance_scale=1.5,     #5
            ).images[0]
            # 保存生成的图片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"generated_image_{timestamp}_step{steps}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            # image.save(output_path)
            # print(f"Image saved at: {output_path}")
            # 将图片加入列表
            images.append(image)
        except Exception as e:
            print(f"Step {steps+1}: Error - {e}")
            continue
        # 如果生成了图片，则保存所有图片到一个网格中
if len(images) > 0:
    print("Creating grid image...")
    try:
        grid_size = int(np.ceil(np.sqrt(len(images))))
        grid_width = grid_size * images[0].width
        grid_height = grid_size * images[0].height
        grid_image = Image.new('RGB', (grid_width, grid_height))

        for idx, img in enumerate(images):
            x = (idx % grid_size) * img.width
            y = (idx // grid_size) * img.height
            grid_image.paste(img, (x, y))

        # 保存网格图片
        grid_output_path = os.path.join(output_dir, f"grid_{timestamp}.jpg")
        grid_image.save(grid_output_path)
        print(f"Grid image saved at: {grid_output_path}")
    except Exception as e:
        print(f"Error creating grid image: {e}")
else:
    print("No images generated, skipping grid creation.")

        # image.save('result.jpg')