import logging
import os
from urllib import request

import torch

from ...ml_decoder.ml_decoder import add_ml_decoder_head

logger = logging.getLogger(__name__)

from ..tresnet import TResnetM, TResnetL, TResnetXL

from ..vit import VE


def create_model(args,load_head=False):
    """Create a model
    """
    model_params = {'args': args, 'num_classes': args.num_classes, 'image_size': args.image_size}
    args = model_params['args']
    args.model_name = args.model_name.lower()

    if args.model_name == 'vit':
        model = VE(model_params)
    elif args.model_name == 'tresnet_m':
        model = TResnetM(model_params)
    elif args.model_name == 'tresnet_l':
        model = TResnetL(model_params)
    elif args.model_name == 'tresnet_xl':
        model = TResnetXL(model_params)
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    ####################################################################################
    if args.use_ml_decoder:
            model = add_ml_decoder_head(model,num_classes=args.num_classes,num_of_groups=args.num_of_groups,
                    decoder_embedding=args.decoder_embedding, zsl=args.zsl)
    ####################################################################################
    # loading pretrain model
    model_path = args.model_path
    if model_path:  # make sure to load pretrained model
        state = torch.load(model_path, map_location='cpu')
        if 'model' in state:
            key = 'model'
        else:
            key = 'state_dict'     
 
        if key == 'model':
            model.load(model_path)
        else:
            new_state_dict = {}
            for k, v in state.items():
                if k.startswith('module.'):
                    name = k[7:]  # 去除 'module.' 前缀
                    new_state_dict[name] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)

    return model
