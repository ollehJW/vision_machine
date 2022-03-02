import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
from skimage import io

transform_dict = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

class ConstructDataset(Dataset):
    """
    Construct pytorch Dataset from file list.

    Parameters
    ----------
    file_list : list
        image file list
    target_list : list
        target list
    phase : str
        train phase. (Default: 'train')

    Returns
    --------
    pytorch Dataset
    """

    def __init__(self, file_list, target_list, phase = 'train'):
        self.file_list = file_list
        self.target_list = target_list
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        target_class = self.target_list[index]
        image = cv2.imread(file_name)
        image = transform_dict[self.phase](image)
        return {'image': image, 'target': target_class}

class dataset_generator(object):
    """
    Construct pytorch DataLoader from file list.

    Parameters
    ----------
    file_list : list
        image file list
    target_list : list
        target list
    batch_size : int
        batch size. (Default: 16)
    phase : str
        train phase. (Default: 'train')
    train_valid_split : bool
        whether to split data with train and validation. (Default: False)  
    valid_ratio : float
        validation ratio. (Default: 0.2) 
    stratify : list  
        Target to be used for stratified extraction (Default: None) 
    random_seed : int
        random seed number (Default: 1004)  

    Returns
    --------
    pytorch DataLoader
    """
    def __init__(self, file_list, target_list, batch_size = 16, phase = 'train', train_valid_split = False, valid_ratio = 0.2, stratify = None, random_seed = 1004):
        self.file_list = file_list
        self.target_list = target_list
        self.batch_size = batch_size
        self.phase = phase
        self.train_valid_split = train_valid_split
        self.valid_ratio = valid_ratio
        self.stratify = stratify
        self.random_seed = random_seed

    def dataloader(self):

        if self.phase == 'train':
            if self.train_valid_split:
                train_file_list, valid_file_list, train_target_list, valid_target_list = train_test_split(self.file_list, 
                                                                                        self.target_list, 
                                                                                        test_size=self.valid_ratio, 
                                                                                        stratify=self.stratify,
                                                                                        random_state=self.random_seed)
                
                train_dataset = ConstructDataset(train_file_list, train_target_list, phase = 'train')
                valid_dataset = ConstructDataset(valid_file_list, valid_target_list, phase = 'valid')

                return dict({'train': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
                            'valid': DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)})


            else:
                train_dataset = ConstructDataset(self.file_list, self.target_list, phase = 'train')
                return dict({'train': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)})
        
        else:
            test_dataset = ConstructDataset(self.file_list, self.target_list, phase = self.phase)
            return dict({'test': DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)})


