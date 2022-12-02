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
#parser.add_argument('--model_checkpoint', type=str, default='D:/phd stuff/retinal_lesion_code/retinal_lesions_code/model_checkpoints/UNet_checkpoints/UNet_epoch_61.pth',
#                    help='dir for model training checkpoint')
parser.add_argument('--num_classes', type=int, default=8, help='number of classes for the model')
parser.add_argument('--train_test_split', type=float, default=0.8, help='ratio of train to test samples')
parser.add_argument('--model_save_directory', type=str, default='D:/phd stuff/retinal_lesion_code/retinal_lesions_code/model_checkpoints/UNet_checkpoints',
                    help='where to save models')
parser.add_argument('--results_save_directory', type=str, default='D:/phd stuff/retinal_lesion_code/retinal_lesions_code/model_checkpoints/UNet_results',
                    help='where to save results')
args = parser.parse_args()

def train_model(model, epochs, batch_size, batches_per_epoch, train_img_paths, train_dataframe, test_img_paths, test_dataframe, criterion, device):

    test_ious, test_f1s, test_precisions = [],[],[]

    batches_per_epoch = int((len(train_img_paths)*5)/batch_size)

    for i in range(epochs):
        print(model.training)
        print(device)
        pbar = tqdm(range(0, batches_per_epoch))#
        epoch_iou = 0
        running_loss = 0
        for j in pbar:
            model.train()
            optimizer.zero_grad()
            batch_indices = np.random.randint(0,len(train_img_paths),batch_size)
            batch_paths = train_img_paths[batch_indices]

            images, gt_masks, _ = prepare_batch(batch_paths, train_dataframe)
            images = images.to(device)
            gt_masks = gt_masks.to(device)
            logits = model(images)
            """softmaxing on the channels"""
            logits = F.softmax(logits * 100, dim=1)


            tp, fp, fn, tn = segmentation_models_pytorch.metrics.get_stats(logits.contiguous(), gt_masks.int().contiguous(), mode='multilabel', threshold=0.5)
            epoch_iou += float(segmentation_models_pytorch.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise").cpu().numpy())
            loss = criterion(logits.contiguous(), gt_masks.contiguous())
            running_loss += loss
            loss.backward()
            optimizer.step()


            pbar.set_description(
                f'Epoch={i}, Train_Loss={running_loss/j}')

        val_iou, val_f1, val_precision = test_model(model, batch_size, test_img_paths, test_dataframe, device)
        test_ious.append(val_iou)
        test_f1s.append(val_f1)
        test_precisions.append(val_precision)

        print(f'Epoch {i} : Train Iou = {epoch_iou/batches_per_epoch} , Val IOU = {val_iou} \n Val F1 = {val_f1}, Val Precision = {val_precision}')
        torch.save(model.state_dict(), args.model_save_directory+f'/UNet_Epoch_{i}.pth')

        iou_np = np.expand_dims(np.array(test_ious), 0)
        f1_np = np.expand_dims(np.array(test_f1s), 0)
        precision_np = np.expand_dims(np.array(test_precisions), 0)
        test_metrics = np.concatenate((iou_np, f1_np, precision_np), axis=0)

        with open(args.results_save_directory+'/'+'test_results.npy', 'wb') as f:
            np.save(f, test_metrics)


def predict_images(model, image, gt_mask):
    model.eval()
    image = image.cpu().numpy()
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image)
    gt_mask = gt_mask.cpu().numpy()
    gt_mask = np.expand_dims(gt_mask, 0)
    gt_mask = torch.from_numpy(gt_mask)
    logits = model(image)
    logits = logits.cpu()
    tp, fp, fn, tn = segmentation_models_pytorch.metrics.get_stats(logits.contiguous(), gt_mask.int().contiguous(),
                                                                   mode='multilabel', threshold=0.5)
    iou = float(segmentation_models_pytorch.metrics.iou_score(tp, fp, fn, tn, reduction='macro-imagewise').cpu().numpy())
    print(iou)
    logits = F.softmax(logits * 100, dim=1)
    logits = logits.permute(0,2,3,1)
    logits = logits.cpu().detach().numpy()
    logits = logits.squeeze()
    return logits

def test_model(model, batch_size, img_paths, dataframe, device):
    model.eval()
    print(model.training)
    index = 0
    iou = 0
    f1 = 0
    precision = 0
    while index < len(img_paths):
        current_batch_size = min(batch_size, len(img_paths)-index)
        current_batch_paths = img_paths[index:index+current_batch_size]
        batch_images, gt_masks, _ = prepare_batch(current_batch_paths, dataframe)

        batch_images = batch_images.to(device)
        gt_masks = gt_masks.to(device)

        logits = model(batch_images)
        """softmaxing on the channels"""
        logits = F.softmax(logits*100, dim=1)

        """make the logits b,h,w,c and do the same for masks"""
        #logits = logits.permute(0, 3, 1, 2)
        #gt_masks = gt_masks.permute(0, 3, 1, 2)

        #window_name = 'image'
        #cv2.imshow(window_name, logits[0])
        #cv2.waitkey(0)
        #cv2.destroyAllWindows()

        tp, fp, fn, tn = segmentation_models_pytorch.metrics.get_stats(logits.contiguous(), gt_masks.int().contiguous(), mode='multilabel',
                                                                       threshold=0.5)
        iou += float(segmentation_models_pytorch.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise").cpu().numpy())
        f1 += float(segmentation_models_pytorch.metrics.f1_score(tp, fp, fn, tn, reduction="macro-imagewise").cpu().numpy())
        precision += float(segmentation_models_pytorch.metrics.precision(tp, fp, fn, tn, reduction="macro-imagewise").cpu().numpy())

        index += current_batch_size

    return iou/len(img_paths), f1/len(img_paths), precision/len(img_paths)




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('preparing model...')
    model = Unet(args.num_classes)

    model = torch.nn.DataParallel(model)
    if args.model_checkpoint != None:
        checkpoint = torch.load(args.model_checkpoint, map_location='cpu')#['state_dict']
        ckpt_filter = {k: v for k, v in checkpoint.items() if 'criterion.0.criterion.weight' not in k}
        model.load_state_dict(ckpt_filter, strict=False)

    model.to(device)
    print('model loaded')

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
    train_model(model, epochs = 100, batch_size = 6, batches_per_epoch = 1000,
                train_img_paths = train_img_paths, train_dataframe = train_dataframe,
                test_img_paths = test_img_paths, test_dataframe = test_dataframe, criterion = criterion,
                device = device)
    """

    test_batch_images, gt_masks,_ = prepare_batch(test_img_paths[:2], test_dataframe)
    logits = predict_images(model, test_batch_images[0], gt_masks[0])
    #print(np.max(logits))
    for i in range(0,logits.shape[-1]):
        cv2.imshow('logits', (logits[:,:,i])/2+(gt_masks[0].permute(1,2,0).cpu().numpy()[:,:,i])/2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """