import torch.nn as nn
import torch
import torch.nn.functional as F
import cv2
import pandas
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
parser.add_argument('--train_test_split', type=float, default=0.8, help='ratio of train to test samples')
parser.add_argument('--model_save_directory', type=str, default='D:/phd stuff/retinal_lesion_code/retinal_lesions_code/model_checkpoints/UNet_checkpoints',
                    help='where to save models')
parser.add_argument('--results_save_directory', type=str, default='D:/phd stuff/retinal_lesion_code/retinal_lesions_code/model_checkpoints/UNet_results',
                    help='where to save results')
args = parser.parse_args()

def train_model(model, epochs, batch_size, batches_per_epoch, train_img_paths, train_dataframe, test_img_paths, test_dataframe, criterion):

    test_ious, test_f1s, test_precisions = [],[],[]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(epochs):
        model.train()
        pbar = tqdm(range(0, batches_per_epoch))#
        epoch_iou = 0
        running_loss = 0
        for j in pbar:
            batch_indices = np.random.randint(0,len(train_img_paths),batch_size)
            batch_paths = train_img_paths[batch_indices]

            images, gt_masks, _ = prepare_batch(batch_paths, train_dataframe)
            images = images.to(device)
            gt_masks = gt_masks.to(device)
            logits = model(images)
            """softmaxing on the channels"""
            logits = F.softmax(logits * 100, dim=1)

            """make the logits b,h,w,c and do the same for masks"""
            logits = logits.permute(0,3,1,2)
            gt_masks = gt_masks.permute(0,3,1,2)


            tp, fp, fn, tn = segmentation_models_pytorch.metrics.get_stats(logits.contiguous(), gt_masks.contiguous(), mode='multilabel', threshold=0.5)
            epoch_iou += float(segmentation_models_pytorch.metrics.iou_score(tp, fp, fn, tn, reduction="micro").cpu().numpy())
            loss = criterion(logits.contiguous(), gt_masks.contiguous())
            running_loss = running_loss * 0.99 + loss * 0.01
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description(
                f'Epoch={i}, Train_Loss={running_loss}')

        val_iou, val_f1, val_precision = test_model(model, batch_size, test_img_paths, test_dataframe)
        test_ious.append(val_iou)
        test_f1s.append(val_f1)
        test_precisions.append(val_precision)

        print(f'Epoch {i} : Train Iou = {epoch_iou/batches_per_epoch} , Val IOU = {val_iou} \n Val F1 = {val_f1}, Val Precision = {val_precision}')
        torch.save(model.state_dict(), args.model_save_directory+f'/UNet_Epoch_{i}')

        iou_np = np.expand_dims(np.array(test_ious), 0)
        f1_np = np.expand_dims(np.array(test_f1s), 0)
        precision_np = np.expand_dims(np.array(test_precisions), 0)
        test_metrics = np.concatenate((iou_np, f1_np, precision_np), axis=0)

        #with open(args.results_save_directory+'/'+'test_results.npy', 'wb') as f:
        #    np.save(f, test_metrics)


def predict_images(image):
    output = F.one_hot(image.argmax(dim=0), image.shape[0])
    return output

def test_model(model, batch_size, img_paths, dataframe):
    model.eval()

    index = 0
    iou = 0
    f1 = 0
    precision = 0
    while index < len(img_paths):
        current_batch_size = min(batch_size, len(img_paths)-index)
        current_batch_paths = img_paths[index:index+current_batch_size]
        batch_images, gt_masks, _ = prepare_batch(current_batch_paths, dataframe)
        logits = model(batch_images)
        """softmaxing on the channels"""
        logits = F.softmax(logits * 100, dim=1).squeeze()

        """make the logits b,h,w,c and do the same for masks"""
        logits = logits.transpose(0, 2, 3, 1)
        gt_masks = gt_masks.transpose(0, 2, 3, 1)

        tp, fp, fn, tn = segmentation_models_pytorch.metrics.get_stats(logits, gt_masks, mode='multilabel',
                                                                       threshold=0.5)
        iou += float(segmentation_models_pytorch.metrics.iou_score(tp, fp, fn, tn, reduction="micro").cpu().numpy())
        f1 += float(segmentation_models_pytorch.metrics.f1_score(tp, fp, fn, tn, reduction="micro").cpu().numpy())
        precision += float(segmentation_models_pytorch.metrics.precision(tp, fp, fn, tn, reduction="macro").cpu().numpy())

    return iou/len(img_paths), f1/len(img_paths), precision/len(img_paths)




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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    """using dice loss for multi-class semantic segmentation, https://arxiv.org/pdf/2006.14822.pdf"""
    criterion = JaccardLoss(mode='multilabel', from_logits=False)


    dataframe = pandas.read_csv(args.dataset_dir + '/dr_grades.csv').values
    all_img_paths = dataframe[:,0].copy()
    for j in range(len(all_img_paths)):
        all_img_paths[j] = args.dataset_dir + '/images_896x896/' + all_img_paths[j] + '.jpg'

    train_img_paths = all_img_paths[:int(len(all_img_paths)*args.train_test_split)]
    test_img_paths = all_img_paths[int(len(all_img_paths)*args.train_test_split):]


    train_dataframe = dataframe[:int(len(all_img_paths)*args.train_test_split)]

    names_list = list(dataframe[:,0])
    #print(names_list)
    #print('all names in dataframe')

    test_dataframe = dataframe[int(len(all_img_paths)*args.train_test_split):]

    train_model(model, epochs = 100, batch_size = 8, batches_per_epoch = 300,
                train_img_paths = train_img_paths, train_dataframe = train_dataframe,
                test_img_paths = test_img_paths, test_dataframe = test_dataframe, criterion = criterion)