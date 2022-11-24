import torch.nn as nn
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss
import segmentation_models_pytorch
import argparse
from models.UNet import Unet
from src.utils import get_paths, prepare_batch

"""training hub, at the moment it just trains the UNet but in future will be used for other models too"""

parser = argparse.ArgumentParser(description='Joseph Muddle\'s semantic segmentation model')
parser.add_argument('--dataset_dir', type=str, default = 'D:/phd stuff/retinal lesions/retinal-lesions-v20191227', help='root dir for the data')
parser.add_argument('--model_checkpoint', type=str, default=None, help='dir for model training checkpoint')
parser.add_argument('--num_classes', type=int, default=8, help='number of classes for the model')
args = parser.parse_args()

def train_model(model, epochs, batch_size, batches_per_epoch):
    all_img_paths = get_paths(args.dataset_dir)
    all_img_paths = np.array(all_img_paths)
    for i in range(epochs):
        pbar = tqdm(range(0, batches_per_epoch))
        for j in pbar:
            batch_indices = np.random.randint(0,len(all_img_paths),batch_size)
            batch_paths = all_img_paths[batch_indices]

            images, masks, _ = prepare_batch(batch_paths)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('preparing model...')
    model = Unet(args.num_classes)

    model = torch.nn.DataParallel(model)
    if args.model_checkpoint != None:
        checkpoint = torch.load(args.model_checkpoint, map_location='cpu')['state_dict']
        ckpt_filter = {k: v for k, v in checkpoint.items() if 'criterion.0.criterion.weight' not in k}
        model.load_state_dict(ckpt_filter, strict=False)

    model.to(device)
    print('model loaded')

    """using Adam optimizer with small learning rate, as we are fine tuning the model """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    """using dice loss for multi-class semantic segmentation, https://arxiv.org/pdf/2006.14822.pdf"""
    criterion = JaccardLoss(mode='multilabel', from_logits=True)



train_model('a',10,10, 10)