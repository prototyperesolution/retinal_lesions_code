import cv2
import os
import glob
import numpy as np

"""utils for the dataset, including loading the images and masks for segmentation and loading the DR grades for 
classification/regression"""

def get_paths(dataset_dir):
    img_dir = dataset_dir + '/images_896x896'
    img_paths = glob.glob(img_dir+'/*.jpg')
    return img_paths

def prep_seg_masks(img_paths, mask_size=896):
    """finds the corresponding segmentation masks to each image and prepares them as one hot vectors"""
    """assuming here that intraretinal hemmorhage as described in the github is replaced by retinal hemmorage"""
    classes = ['microaneurysm', 'retinal_hemorrhage', 'hard_exudate', 'cotton_wool_spots',
               'vitreous_hemorrhage', 'preretinal_hemorrhage', 'neovascularization', 'fibrous_proliferation']
    masks = np.zeros((len(img_paths), mask_size, mask_size, len(classes)))
    for i in range(len(img_paths)):

        img_dir, name = os.path.split(img_paths[i])
        name = name[:-4]
        root_dir = os.path.split(img_dir)[0]
        mask_dir = root_dir + '/lesion_segs_896x896/'+name
        mask_paths = glob.glob(mask_dir+'/*.png')

        current_mask = np.zeros((mask_size, mask_size, len(classes)))
        for mask_path in mask_paths:
            img = cv2.imread(mask_path)
            one_hot = (img/255)[:,:,0]
            _, name = os.path.split(mask_path)
            name = name[:-4]
            """inserting current one hot into proper position in current mask"""
            current_mask[:,:,classes.index(name)] = one_hot
        masks[i] = current_mask

    return masks



def prepare_semantic_batch(img_paths):
    print(os.path.split(img_paths[0]))

img_paths = get_paths('D:/phd stuff/retinal lesions/retinal-lesions-v20191227')[:10]
test_masks = prep_seg_masks(img_paths)
