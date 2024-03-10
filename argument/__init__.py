#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group
    
    def load_yaml(self, opts=None):
        if opts is None:
            return
        else:
            for key, value in opts.items():
                try:
                    setattr(self, key, value)
                except:
                    raise Exception(f'Unknown attribute {key}')


class TrainEmbeddingParams(ParamGroup):
    def __init__(self, parser, opts=None):
        '''id embeddings'''
        saved_embeds_path = None
        save_steps = 100
        only_save_embeds = True
        placeholder_token = "<28017>"
        train_batch_size = 8
        scale_lr = True
        n_persudo_tokens = 2
        learning_rate = 0.000625
        max_train_steps = 320
        train_data_dir = "id_images/28017"
        celeb_path = "cross_initialization/examples/wiki_names_v2.txt"
        pretrained_model_name_or_path = "/home/yhe/projects/zeyu/weights/huggingface/base_model/stable-diffusion-2-1-base"
        output_dir = "./logs/28017/id_embeddings"

        super().__init__(parser, "TrainEmbedding Parameters")



class GuidanceParams(ParamGroup):
    def __init__(self, parser, opts=None):
        fp16 = False
        base_model_path = "/home/yhe/projects/zeyu/weights/huggingface/base_model/stable-diffusion-2-1-base"
        id_embeds_path = None
        controlnet_path = None
        seed = 1234
        vram_O = False
        min_range = 0.02
        max_range = [0.5, 0.98]
        anneal_timesteps = True
        anneal_strategy = 'linear'
        total_iteration = 1000
        guidance_scale = 7.5
        consistent_embeds_lr = 0.0006
        initial_consistent_prompt = "unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, dark, low-resolution, gloomy"



def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)