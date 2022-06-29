from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms as T
from metrics import StreamSegMetrics

import sys
sys.path.insert(0, '../..')
from data import *
from datasets import *

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from glob import glob

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
   # parser.add_argument("--input", type=str, required=True,
                        #help="path to a single image or image directory")
    parser.add_argument("--dataset", type=str, default='synthia',
                        choices=['voc', 'cityscapes'], help='Name of training set')
    parser.add_argument("--test_num", type=int, default=1)

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default=None,
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    
    parser.add_argument("--ckpt", default='checkpoints/best_deeplabv3plus_resnet50_synthia_os16.pth', type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target
    elif opts.dataset.lower() == 'synthia':
        opts.num_classes = 23
       # decode_fn = Synthia.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    target_loader = setup_loaders(['synthia'], ['test.txt'], 1)
    targetloader_iter = enumerate(target_loader)
    
    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)
        
    with torch.no_grad():
        model = model.eval()
        for i in (range(opts.test_num)):
            _, batch = targetloader_iter.__next__()
            img, mask = batch
            img = img.to(device, dtype=torch.float32)
            mask = mask.to(device, dtype=torch.long)
            
            outputs = model(img)
            pred = outputs.detach().max(dim=1)[1].cpu().numpy()
            mask = mask.cpu().numpy()
            
            img, mask, pred = img[0], mask[0], pred[0]

            fig = plt.figure(figsize=(19.2, 10.8))
            fig.add_subplot(1, 3, 1)
            plt.title('Image')
            trans = T.ToPILImage()
            torch.as_tensor(img).float().contiguous()
            plt.imshow(trans(img))

            cmap = 'tab20' #had tab20, nipy_spectral, gnuplot
            
            fig.add_subplot(1, 3, 2)
            plt.title('Mask GT')
            values = np.unique(mask.ravel())
            names = get_label_names()
            im = plt.imshow(mask, cmap=cmap) 
            colors = [ im.cmap(im.norm(value)) for value in values]
            patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=names[i]) ) for i in range(len(values)) ]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

            fig.add_subplot(1, 3, 3)
            plt.title('Pred')
            values = np.unique(pred.ravel())
            im = plt.imshow(pred, cmap=cmap)
            colors = [ im.cmap(im.norm(value)) for value in values]
            patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=names[values[i]]) ) for i in range(len(values)) ]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
            
            plt.subplots_adjust(wspace=0.6)
            
            plt.show()
            
            
def get_label_names():
    with open('./dataset/synthia_3channels.yaml', 'r') as stream:
        synthiayaml = yaml.safe_load(stream)
    return synthiayaml['name']


if __name__ == '__main__':
    main()