import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torchvision.utils import save_image

from tqdm import tqdm
from torch.cuda.amp import custom_bwd, custom_fwd
from diffusers import (StableDiffusionPipeline, AutoencoderKL, DDIMScheduler,
                       DDPMScheduler, ControlNetModel)
from transformers import CLIPTokenizer, CLIPTextModel
import cv2


def get_generator(seed, device):

    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator


class StableDiffusionDualEmbedsGuidance:
    def __init__(self, device, config, id_embeds_path, use_controlnet=False) -> None:
        self.device = device
        self.config = config
        self.torch_dtype = torch.float16 if config.fp16 else torch.float32

        self.base_model_path = config.base_model_path
        self.id_embeds_path = id_embeds_path
        self.controlnet_path = config.controlnet_path

        self.seed = config.seed
        self.generator = get_generator(self.seed, self.device)

        self.use_controlnet = use_controlnet

        self.vram_O = config.vram_O

        '''load id embeddings to text_encoder and tokenizer'''
        tokenizer, text_encoder, tokens = self.load_id_embedings(
            self.base_model_path,
            self.id_embeds_path,
        )

        self.tokens = tokens

        pipe = StableDiffusionPipeline.from_pretrained(
            self.base_model_path,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            torch_dtype=self.torch_dtype,
        ).to(self.device)

        if self.vram_O:
            self.pipe.enable_sequential_cpu_offload()
            self.pipe.enable_vae_slicing()
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe.enable_attention_slicing(1)
            self.pipe.enable_model_cpu_offload()

        self.pipe = pipe
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.use_controlnet = use_controlnet

        if self.use_controlnet:
            self.controlnet = ControlNetModel.from_pretrained(
                self.controlnet_path,
                torch_dtype=self.torch_dtype
            ).to(self.device)
        else:
            self.controlnet = None

        '''timestep related''' 
        self.scheduler = DDPMScheduler.from_config(
            pipe.scheduler.config
        )
        # self.scheduler = DDIMScheduler.from_config(
        #     pipe.scheduler.config
        # )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.timesteps = torch.flip(self.scheduler.timesteps, dims=(0, ))
        min_range = config.min_range
        # [0.5, 0.98]
        max_range = config.max_range
        self.min_step = int(self.num_train_timesteps * min_range)
        self.max_step = int(self.num_train_timesteps * max_range[1])
        self.max_step_0 = int(self.num_train_timesteps * max_range[0])
        self.max_step_1 = int(self.num_train_timesteps * max_range[1])
        
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        self.anneal_timesteps = config.anneal_timesteps
        self.anneal_strategy = config.anneal_strategy
        self.total_iteration = config.total_iteration

        self.guidance_scale = config.guidance_scale

        # consistent embeds
        self.initial_consistent_prompt = config.initial_consistent_prompt
        self.consistent_embeds = self.get_text_embeds(prompt=self.initial_consistent_prompt)
        # self.learnable_embedding = torch.nn.Parameter(torch.zeros((77, 1024)).to(self.device))
        self.consistent_embeds.requires_grad = True

        print(f'[INFO] loaded guidance (stable diffusion + customized embeddings)!')


    def load_id_embedings(self, pretrained_model_name_or_path, learned_embed_name_or_path):
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, 
            subfolder="tokenizer", 
            torch_dtype=self.torch_dtype
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, 
            subfolder="text_encoder", 
            torch_dtype=self.torch_dtype
        ).to(self.device)

        embeds_dict=torch.load(learned_embed_name_or_path)
        tokens=list(embeds_dict.keys())
        embeds = [embeds_dict[token]for token in tokens]

        tokenizer.add_tokens(tokens)
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        for i, token_id in enumerate(token_ids):
            text_encoder.get_input_embeddings().weight.data[token_id] = embeds[i]

        return tokenizer, text_encoder, tokens
    
    @torch.no_grad()
    def get_text_embeds_cond(self, prompt):
        prompt=prompt.format(" ".join(self.tokens))
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings    
    
    def decode_latents(self, latents):
        target_dtype = latents.dtype
        latents = latents / self.vae.config.scaling_factor

        imgs = self.vae.decode(latents.to(self.vae.dtype)).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs.to(target_dtype)
    
    def adjust_max_step(self, iteration):
        max_step_0 = self.max_step_0
        max_step_1 = self.max_step_1

        now_max_step = max_step_1 if (iteration / self.total_iteration) < 0.5 else max_step_0

        return now_max_step
    
    # @torch.no_grad()
    def encode_imgs(self, imgs):
        target_dtype = imgs.dtype
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs.to(self.vae.dtype)).latent_dist
        kl_divergence = posterior.kl()

        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents.to(target_dtype), kl_divergence

    def get_anneal_ind_t(self, iteration, mode='hifa'):
        mode = self.anneal_strategy
        if mode == 'hifa':
            ind_t = int(self.max_step - (self.max_step - self.min_step) * math.sqrt(iteration / self.total_iteration))
        elif mode == 'linear':
            ind_t = int(self.max_step - (self.max_step - self.min_step) * (iteration / self.total_iteration))
        else:
            raise Exception("Not Implemented Time Annealing Strategy")
        
        return ind_t

    def train_consistent_embeds(
        self,
        text_embeds, # uncond cond
        pred_rgb,
        pred_skeleton=None,
        iteration=None,
        resolution=(512,512),
    ):
        if pred_rgb.shape[1] != 4:
            pred_rgb = F.interpolate(pred_rgb, resolution, mode='bilinear', align_corners=False)
            # pred_skeleton = F.interpolate(pred_skeleton, resolution, mode='bilinear', align_corners=False)
            img_latents, _ = self.encode_imgs(pred_rgb)
        else:
            img_latents = pred_rgb

        if self.anneal_timesteps:
            ind_t = self.get_anneal_ind_t(iteration=iteration)
        else:  
            self.max_step = self.adjust_max_step(iteration)
            ind_t = torch.randint(self.min_step, self.max_step, (pred_rgb.shape[0],), dtype=torch.long, device=self.device)[0]
            
        t = int(self.timesteps[ind_t])
        t = torch.tensor([t], dtype=torch.long, device=self.device)

        # if not self.changed:
        #     self.consistent_embeds.data = text_embeds[None, 0]
        #     self.changed = True

        consistent_embeds = self.consistent_embeds.repeat(img_latents.shape[0], 1, 1)

        noise = torch.randn(img_latents.shape, generator=self.generator, dtype=self.torch_dtype, device=self.device)
        latents_noisy = self.scheduler.add_noise(img_latents, noise, t)
        latent_model_input = latents_noisy[None, :, ...].repeat(1, 1, 1, 1, 1).reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
        tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)

        down_block_res_samples = mid_block_res_sample = None

        if self.use_controlnet:
            pred_skeleton_input = pred_skeleton.repeat(latent_model_input.shape[0], 1, 1, 1).reshape(-1, 3, resolution[0], resolution[1]).to(self.torch_dtype)
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                tt,
                encoder_hidden_states=consistent_embeds,
                controlnet_cond=pred_skeleton_input,
                return_dict=False,
            )
        noise_pred = self.unet(
            latent_model_input,
            tt,
            encoder_hidden_states=consistent_embeds,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample, 
        ).sample

        loss = 0.5 * F.mse_loss(noise_pred, noise.detach(), reduction="mean") / img_latents.shape[0]

        return loss

    def train_sds(
        self,
        text_embeds, # cond uncond
        pred_rgb,
        pred_skeleton=False,
        iteration=None,
        resolution=(512,512),
        save_folder=None,
        vis_interval=20,
    ):
        if pred_rgb.shape[1] != 4:
            pred_rgb = F.interpolate(pred_rgb, resolution, mode='bilinear', align_corners=False)
            # pred_skeleton = F.interpolate(pred_skeleton, resolution, mode='bilinear', align_corners=False)
            img_latents, _ = self.encode_imgs(pred_rgb)
        else:
            img_latents = pred_rgb

        if self.anneal_timesteps:
            ind_t = self.get_anneal_ind_t(iteration=iteration)
        else:  
            self.max_step = self.adjust_max_step(iteration)
            ind_t = torch.randint(self.min_step, self.max_step, (pred_rgb.shape[0],), dtype=torch.long, device=self.device)[0]

        t = int(self.timesteps[ind_t])
        t = torch.tensor([t], dtype=torch.long, device=self.device)

        # use negative_prompt ? if used, the results maybe better but the speed may be a little bit slower
        uncond_embeds, cond_embeds = text_embeds.chunk(2)
        consistent_embeds = self.consistent_embeds.repeat(img_latents.shape[0], 1, 1).detach()

        input_embeds = torch.cat([consistent_embeds, uncond_embeds, cond_embeds], dim=0)

        with torch.no_grad():
            noise = torch.randn(
                img_latents.shape,
                dtype=self.torch_dtype, device=self.device, generator=self.generator
            )   
            latents_noisy = self.scheduler.add_noise(img_latents, noise, t)
            latent_model_input = latents_noisy[None, :, ...].repeat(3, 1, 1, 1, 1).reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
            tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)

            down_block_res_samples = mid_block_res_sample = None
            
            if self.use_controlnet:
                pred_skeleton_input = pred_skeleton.repeat(latent_model_input.shape[0], 1, 1, 1).reshape(-1, 3, resolution[0], resolution[1]).to(self.torch_dtype)
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    tt,
                    encoder_hidden_states=input_embeds,
                    controlnet_cond=pred_skeleton_input,
                    return_dict=False,
                )

            noise_pred = self.unet(
                latent_model_input,
                tt,
                encoder_hidden_states=input_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample, 
            ).sample

            noise_pred_consistent, noise_pred_uncond, noise_pred_text = noise_pred.chunk(3)

            delta_cond = self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            if t < 200:
                delta_uncond = noise_pred_text
            else:
                delta_uncond = noise_pred_text - noise_pred_consistent

        noise_pred = delta_uncond + delta_cond

        w = (1 - self.alphas[t.item()]).view(-1, 1, 1, 1)

        grad = w * (noise_pred)
        grad = torch.nan_to_num(grad)

        loss = 0.5 * F.mse_loss(img_latents, (img_latents - grad).detach(), reduction="mean") / img_latents.shape[0]
        
        if iteration % vis_interval == 0:
            save_path_iter = os.path.join(save_folder,"iter_{}_step_{}.jpg".format(iteration, t.item()))
            with torch.no_grad():    
                labels = self.decode_latents((img_latents - grad).type(self.torch_dtype))

                grad_abs = torch.abs(grad.detach())
                norm_grad  = F.interpolate((grad_abs / grad_abs.max()).mean(dim=1,keepdim=True), (resolution[0], resolution[1]), mode='bilinear', align_corners=False).repeat(1,3,1,1)

                pred_rgb = self.decode_latents(img_latents.detach())
                viz_images = torch.cat([pred_rgb, norm_grad,
                                        labels],dim=0) 
                save_image(viz_images, save_path_iter)

        return loss

    def train_step(
        self,
        text_embeds, # cond uncond
        pred_rgb,
        pred_skeleton=None,
        iteration=None,
        resolution=(512,512),
        save_folder=None,
        vis_interval=20,
    ):
        loss_embeds = self.train_consistent_embeds(
            text_embeds=text_embeds,
            pred_rgb=pred_rgb,
            pred_skeleton=pred_skeleton,
            iteration=iteration,
            resolution=resolution,
        )

        loss_sds = self.train_sds(
            text_embeds=text_embeds,
            pred_rgb=pred_rgb,
            pred_skeleton=pred_skeleton,
            iteration=iteration,
            resolution=resolution,
            save_folder=save_folder,
            vis_interval=vis_interval,
        )

        return loss_embeds + loss_sds