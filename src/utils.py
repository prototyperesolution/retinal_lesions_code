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



def prepare_batch(img_paths, dataframe, masks=True, dr_grading=True, img_size = 896):
    """returns a batch for training. This batch includes both semantic masks and DR gradings. One can be ignored if necessary"""
    """going to experiment with using both the author's gradings and the kaggle gradings, then maybe some combo of the two"""
    gradings = np.zeros((len(img_paths),1))
    masks = prep_seg_masks(img_paths, img_size)
    images = np.zeros((len(img_paths), img_size, img_size, 3))

    for i in range(len(img_paths)):
        image = cv2.imread(img_paths[i])
        if image.shape != img_size:
            image = cv2.resize(image, (img_size, img_size))
        print(image.shape)
        """cv2 defaults to reading images in BGR format"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        """normalising to between 0 and 1"""
        image = image / 255
        images[i] = image





img_paths = get_paths('D:/phd stuff/retinal lesions/retinal-lesions-v20191227')[:10]
#test_masks = prep_seg_masks(img_paths)
#print(test_masks.shape)
prepare_batch(img_paths, 'a')

