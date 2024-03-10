from .models.clip_model import CLIPTextModel
from transformers import CLIPTokenizer
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
import random
from typing import Union

@torch.no_grad()
def celeb_names_cross_init(
        celeb_path : str,
        tokenizer : CLIPTokenizer,
        text_encoder: CLIPTextModel,
        n_column: int=2,
    ):
    with open(celeb_path, 'r') as f:
        celeb_names=f.read().splitlines()
    # get embeddings
    col_embeddings=[[]for _ in range(n_column)]
    for name in tqdm(celeb_names,desc='get embeddings'):
        token_ids=tokenizer(
            name,
            padding="do_not_pad",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0] # (n,)
        embeddings = text_encoder.get_input_embeddings().weight.data[token_ids] # (n,1024)

        # remove the start and end characters
        for i in range(1,min(embeddings.shape[0]-1,n_column+1)):
            col_embeddings[i-1].append(embeddings[i].unsqueeze(0))
    # mean for all names
    for i in range(n_column): 
        col_embeddings[i]=torch.cat(col_embeddings[i]).mean(dim=0).unsqueeze(0)
    col_embeddings=torch.cat(col_embeddings) #(n,1024)
    bos_embed,eos_embed,pad_embed=text_encoder.get_input_embeddings().weight.data[[tokenizer.bos_token_id,tokenizer.eos_token_id,tokenizer.pad_token_id]]
    input_embeds=torch.cat([bos_embed.unsqueeze(0),col_embeddings,eos_embed.unsqueeze(0),pad_embed.repeat(75-col_embeddings.shape[0],1)]) # (77,1024)
    # cross init
    col_embeddings=text_encoder(inputs_embeds=input_embeds.unsqueeze(0))[0][0][1:1+n_column] # (n,1024)

    return col_embeddings # (n,1024)

@torch.no_grad()
def token_cross_init(
    tokens : Union[str,list[str]],
    tokenizer : CLIPTokenizer,
    text_encoder: CLIPTextModel,
    return_first_embeds:bool=False,
):
    if isinstance(tokens,list):
        tokens=' '.join(tokens)
    
    token_ids=tokenizer(
        tokens,
        padding="do_not_pad",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(text_encoder.device) # (1,k)
    if return_first_embeds:
        embeds=text_encoder.get_input_embeddings().weight.data[token_ids[0]] # (k+2,1024)
    else:
        embeds=text_encoder(token_ids)[0][0] # (k+2,1024)
    return embeds[1:-1] #(k,1024)

@torch.no_grad()
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

'''
python train_cross_init.py \
    --save_steps 100 \
    --only_save_embeds \
    --placeholder_token "<28017>" \
    --train_batch_size 8 \
    --scale_lr \
    --n_persudo_tokens 2 \
    --reg_weight "1e-5" \
    --learning_rate 0.000625 \
    --max_train_step 320 \
    --train_data_dir "./examples/input_images/28017" \
    --celeb_path "./examples/wiki_names_v2.txt" \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" \
    --output_dir "./logs/28017/learned_embeddings" 
'''

def load_train_config(config, args):
    args.save_steps = config.save_steps
    args.only_save_embeds = config.only_save_embeds
    args.placeholder_token = config.placeholder_token
    args.train_batch_size = config.train_batch_size
    args.scale_lr = config.scale_lr
    args.n_persudo_tokens = config.n_persudo_tokens
    args.reg_weight = config.reg_weight
    args.learning_rate = config.learning_rate
    args.max_train_step = config.max_train_step
    args.train_data_dir = config.train_data_dir
    args.celeb_path = config.celeb_path
    args.pretrained_model_name_or_path = config.pretrained_model_name_or_path
    args.output_dir = config.output_dir
    
    return args