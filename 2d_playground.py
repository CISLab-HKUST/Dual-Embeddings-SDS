import torch
import numpy as np
import os
import sys
import yaml
from argparse import ArgumentParser, Namespace
from torchvision import transforms
from tqdm import tqdm
from torch.optim import Adam
from utils.images_to_video import imgs_to_video
from PIL import Image
from argument import TrainEmbeddingParams, GuidanceParams
from guidance.sd_utils import StableDiffusionDualEmbedsGuidance
from train_cross_init import train
from diffusers.utils import load_image

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--opt', type=str, default="configs/sample.yaml")
    parser.add_argument('--device', type=str, default="cuda")

    train_embeds_config = TrainEmbeddingParams(parser)
    guidance_config = GuidanceParams(parser)

    args = parser.parse_args(sys.argv[1:])

    if args.opt is not None:
        with open(args.opt) as f:
            opts = yaml.load(f, Loader=yaml.FullLoader)
        train_embeds_config.load_yaml(opts.get('TrainEmbeddingParams', None))
        guidance_config.load_yaml(opts.get('GuidanceParams', None))

    if train_embeds_config.saved_embeds_path is None:
        print("\033[0;31;40m", 'Now Training the ID Embeddings', "\033[0m")
        train(train_embeds_config)
        id_embeds_path = os.path.join(train_embeds_config.output_dir, "learned_embeds.bin")
    else:
        id_embeds_path = train_embeds_config.saved_embeds_path
    guidance_config.id_embeds_path = id_embeds_path

    device = torch.device(args.device)
    torch_dtype = torch.float16 if guidance_config.fp16 else torch.float32
    save_folder = "./exprs/test1"
    total_iteration = 1500
    batch_size = 2

    images_save_folder = os.path.join(save_folder, "images")
    os.makedirs(images_save_folder, exist_ok=True)
    video_save_folder = os.path.join(save_folder, "video")
    os.makedirs(video_save_folder, exist_ok=True)

    guidance_config.total_iteration = total_iteration

    guidance = StableDiffusionDualEmbedsGuidance(
        device=device, 
        config=guidance_config, 
        id_embeds_path=id_embeds_path,
    )

    prompt = "a photo of a {} person wearing Wonder Woman's suit"
    negative_prompt = "unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, dark, low-resolution, gloomy"

    cond_embeds = guidance.get_text_embeds_cond(prompt).repeat(batch_size, 1, 1)
    uncond_embeds = guidance.get_text_embeds(negative_prompt).repeat(batch_size, 1, 1)

    text_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0)
    
    # image_tensor = torch.full((batch_size, 3, 512, 512), 0.5).to(dtype=torch_dtype, device=device)
    image = load_image("images/smpl.png").convert("RGB").resize((512, 512))
    transf = transforms.ToTensor()
    image_tensor = transf(image).unsqueeze(0).to(dtype=torch_dtype, device=device).repeat(batch_size, 1, 1, 1)
    image_latents, _ = guidance.encode_imgs(image_tensor)
    image_latents.requires_grad = True
    
    params = []
    params.append({'params': image_latents, 'lr': 1e-2})
    params.append({'params': guidance.consistent_embeds, 'lr': guidance_config.consistent_embeds_lr})

    optimizer = Adam(params, lr=1e-2)

    for iteration in tqdm(range(total_iteration)):
        loss = guidance.train_step(
            text_embeds=text_embeds, 
            pred_rgb=image_latents,
            iteration=iteration,
            resolution=(512, 512),
            save_folder=images_save_folder,
            vis_interval=20,
        )
        # loss = loss_embeds + loss_sds

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    imgs_to_video(images_save_folder, os.path.join(video_save_folder, "videos.mp4"))

