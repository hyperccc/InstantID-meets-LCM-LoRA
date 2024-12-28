import cv2
import torch
import numpy as np
from PIL import Image

from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid_full import StableDiffusionXLInstantIDPipeline, draw_kps

from controlnet_aux import MidasDetector
from diffusers import LCMScheduler
import os
from datetime import datetime
import time




def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

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
    controlnet_depth_path = f'diffusers/controlnet-depth-sdxl-1.0-small'
    
    # Load depth detector
    midas = MidasDetector.from_pretrained("lllyasviel/Annotators")

    # Load pipeline
    controlnet_list = [controlnet_path, controlnet_depth_path]
    controlnet_model_list = []
    for controlnet_path in controlnet_list:
        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
        controlnet_model_list.append(controlnet)
    controlnet = MultiControlNetModel(controlnet_model_list)
    
    base_model_path = 'stabilityai/stable-diffusion-xl-base-1.0'

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
    lcm_lora_path = "./checkpoints/pytorch_lora_weights.safetensors"
    pipe.load_lora_weights(lcm_lora_path)
    pipe.fuse_lora()
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # num_inference_steps = 10
    # guidance_scale = 0.3

    # Infer setting
    # prompt = "breathtaking, award-winning, professional, highly detailed"
    # n_prompt = "ugly, deformed, noisy, blurry, distorted, grainy"

    # prompt="Stacked papercut art. 3D, layered, dimensional, depth, precision cut, stacked layers, papercut, high contrast"
    # n_prompt= "2D, flat, noisy, blurry, painting, drawing, photo, deformed"
    # prompt= "cinematic film still {prompt} . shallow depth of field, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy"
    # n_prompt= "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"
    # prompt="line art drawing . professional, sleek, modern, minimalist, graphic, line art, vector graphics"
    # n_prompt="anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic"
    # n_prompt= "cross-eyed, anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, watermark, painting, drawing, illustration, glitch,deformed, mutated"
    # #prompt="a photography of a man standing in a street lit by lamps, leica 35mm summilux"
    # prompt = "Happy donald trump standing in a beautiful field of flowers, colorful flowers everywhere, perfect lighting, leica summicron 35mm f2.0, kodak portra 400, film grain"
    n_prompt= "cross-eyed, anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch"
    prompt="iconoclastic surreal photo of beautiful supermodel 25 year old white Virgin Mary, kind huge laughing smile and staring intensely straight ahead directly into camera, wearing sparkling crown of pink roses, long wavy black hair, brilliant icy blue eyes centred towards me, wearing long full sparkling white veil covering her hair and queenly pink diamond sparkly modest robes with high neckline, NO CHEST SHOWING symmetrical, iconic, striking, award winning photo, powerful, moving, editorial photoshoot by Alessio Albi, background is raining sparkling hearts in morning flower sunshine. Film Grain."
    face_image = load_image("./examples/IU.png")
    face_image = resize_img(face_image)

    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
    face_emb = face_info['embedding']

    # use another reference image
    pose_image = load_image("./examples/poses/pose_MonaLisa.png")
    pose_image = resize_img(pose_image)

    face_info = app.get(cv2.cvtColor(np.array(pose_image), cv2.COLOR_RGB2BGR))
    pose_image_cv2 = convert_from_image_to_cv2(pose_image)
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
    face_kps = draw_kps(pose_image, face_info['kps'])

    width, height = face_kps.size

    # use depth control
    processed_image_midas = midas(pose_image)
    processed_image_midas = processed_image_midas.resize(pose_image.size)
    
    # enhance face region
    control_mask = np.zeros([height, width, 3])
    x1, y1, x2, y2 = face_info["bbox"]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    control_mask[y1:y2, x1:x2] = 255
    control_mask = Image.fromarray(control_mask.astype(np.uint8))

    #unload LoRA weights
    pipe.unload_lora_weights()
    pipe.scheduler = type(original_scheduler).from_config(original_scheduler_config)
    output_dir = "/data4/students/user24215459/InstantID-main/results"
    os.makedirs(output_dir, exist_ok=True)

    images = []
    for steps in (1, 4, 6, 8, 15, 20, 25, 30, 50):
        print(f"Step {steps+1}: Starting generation...")
        try:
            #generator = torch.Generator(device=pipe.device).manual_seed(1337)
            start_gen = time.time()
            image = pipe(
                prompt=prompt,
                negative_prompt=n_prompt,
                image_embeds=face_emb,
                control_mask=control_mask,
                image=[face_kps, processed_image_midas],   #face+pose
                controlnet_conditioning_scale=[0.8,0.8],
                ip_adapter_scale=0.8,
                num_inference_steps=steps + 1,
                guidance_scale=1.5,
                #generator=generator,   
            ).images[0]
            gen_time = time.time() - start_gen
            print(f"图像生成时间: {gen_time:.2f}s")
            # print(f"Step {steps+1}: Image generated successfully.")
            # # 保存生成的图片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"generated_image_{timestamp}_step{steps}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            # image.save(output_path)
            # print(f"Image saved at: {output_path}")

            # 将图片加入列表 
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