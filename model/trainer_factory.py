from __future__ import absolute_import
from __future__ import print_function

import os
import time
import numpy as np
import torch

try:
    from ..common_utils import logging_utils
except:
    from common_utils import logging_utils


class SupervisedTraining(object):
    """Trainer for Supervised Learning

    Parameters
    ----------
    epoch : int
        max epoch. Defaults to 1000.
    batch_size : int
        training batch size. Defaults to 32.
    result_model_path : str 
        save directory of result model. Defaults to ''.
    eval_metric : str
        Evaluation metric during training. Defaults to 'val_loss'.
    auto_resume_flag : bool
        Flag for automatic resume. Defaults to False.
    checkpoint_flag : bool 
        Flag for checkpoint. Defaults to True.
    early_stopping_flag : bool
        Flag for Early stopping. Defaults to True.
    early_stopping_min_delta : float
        minimum delta threshold when apply early stopping,
        if metric change between previous and current epoch is less than early_stopping_min_delta, 
        training status is not updated. Defaults to 0.0.
    early_stopping_patience : int
        if update not happen in early_stopping_patience, stop training
        if update happen in early_stopping_patience, update the updating counts to zero. Defaults to 5.
    lr_scheduler_name : str
        the name of learning rate scheduler. Defaults to 'constant'.
    lr_scheduler_max_lr : float
        maximum learning rate when apply learning rate scheduler. Defaults to 0.1.
    """
    def __init__(self, 
                epoch=1000, 
                result_model_path='', 
                ):

        # about training strategy
        self.epoch = epoch
        self.result_path = result_model_path
        
        
        # evaluation handling when OOM issue
        self.is_out_of_memory = False
        self._train_data_iterator=None
        self._val_data_iterator=None
        
        # initialize history
        self.history = None


     # main function of train, **train loop
    def train(self, model, train_dataloader, val_dataloader, criterion, optimizer, gpu = True):
        """
        Conduct supervised learning

        Parameters
        ----------
        model
            pytorch-base model
        train_dataloader
            train dataloader
        val_dataloader
            validation dataloader

        Returns
        --------

        """

        # update instance variables
        self.model = model 
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.gpu = gpu
        self.criterion = criterion
        self.optimizer = optimizer
        
        # update model 
        start_train_time = time.time()
        self._model = model
        model_loaded_time = time.time()

        # gpu setting
        if gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self._model.to(device)

        else:
            device = torch.device("cpu")

        min_valid_loss = np.inf

        # training
        
        for e in range(self.epoch):
            train_loss = 0.0
            correct = 0
            total = 0
            model.train()     # Optional when not using Model Specific layer
            for data in self.train_dataloader:
                if torch.cuda.is_available():
                    images, labels = data['image'].float().to(device), data['target'].float().to(device)
        
                optimizer.zero_grad()
                target = model(images)
                loss = criterion(target,torch.argmax(labels, dim=1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = torch.max(target, 1)
                correct += (predicted == torch.argmax(labels, dim=1)).float().sum()
                total += labels.size(0)
            
            train_acc = correct / total
            correct = 0
            total = 0
            valid_loss = 0.0
            model.eval()     # Optional when not using Model Specific layer
            for data in self.val_dataloader:
                if torch.cuda.is_available():
                    data, labels = data['image'].float().to(device), data['target'].float().to(device)
        
                target = model(data)
                loss = criterion(target,torch.argmax(labels, dim=1))
                valid_loss = loss.item() * data.size(0)
                _, predicted = torch.max(target, 1)
                correct += (predicted == torch.argmax(labels, dim=1)).float().sum()
                total += labels.size(0)
            valid_acc = correct / total

            print(f'Epoch {e+1} \t Training Loss: {train_loss / len(train_dataloader)} \t Training Acc: {train_acc} \t\t Validation Loss: {valid_loss / len(val_dataloader)} \t Validation Acc: {valid_acc}')
            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss
                # Saving State Dict
                torch.save(model.state_dict(), self.result_path + 'Best_model.pth')

