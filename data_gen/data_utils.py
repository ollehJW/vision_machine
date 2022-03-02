import numpy as np
import pandas as pd
import os

def prepare_data(train_path, remove_filename_list=[]):
    """
    Extract file names with target class

    Parameters
    ----------
    train_path : str
        train data path (csv or folders with class name)
    remove_filename_list : list
        file names which should be removed.

    Returns
    --------
    file names, unique label names, target class
    """

    if 'csv' == train_path.split('.')[-1]:
        files_df = pd.read_csv(train_path)
        files = files_df['path'].to_list()
        labels = files_df['label'].to_list()
    else:
        files = []
        for ty in os.listdir(train_path):
            filelist = os.listdir(os.path.join(train_path, ty))
            for i, file in enumerate(filelist):
                if file not in remove_filename_list:
                    files.append(os.path.join(train_path, ty, file))

        labels = [file.split('/')[-2] for file in files]

    uni_label = np.unique(labels)
    print("There are {} classes: {}".format(len(uni_label), uni_label))
    y = np.array([np.eye(len(uni_label))[np.where(uni_label==label)].reshape(-1) for label in labels])

    return files, uni_label, y

