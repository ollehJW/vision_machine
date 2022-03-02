import os
import sys
import json
import time
import math
import glob
import numpy as np
import torch
from data_gen.data_utils import prepare_data
from data_gen.data_gen import dataset_generator
from model.loss_factory import LossFactory
from model.model_factory import ModelFactory
from model.optimizer_factory import OptimizerFactory
from model.trainer_factory import SupervisedTraining
from common_utils import logging_utils
from common_utils.common_utils import parse_kwargs
from inference.tester import Tester


class Vision_Machine(object):
    """ A tool to easily conduct for Vision classification
    
    Parameters
    -----------
    cfg_path : str
        json type config path
    
    Examples
    """
    def __init__(self, cfg_path):

        # initialize parameter groups
        self.control_config = {}
        self.parameter_config = {}

        # load json config, parameter groups will have values
        self._load_config(cfg_path)

        # initiate logging
        logging_utils.initiate_log(self.control_config['log_path'])

    def train(self):
        """Vision Machine's training function.
        Parameters should be defined with config path (json type)
        when class instantiated.
            
        Returns
        -----------
        float
            best_value with decided eval metric, 
        str
            result_model_path
        """

        # find file list
        file_list, uni_label, target_list = prepare_data(train_path = self.control_config['train_data_dir'],
                                        remove_filename_list=self.control_config["remove_filename_list"])

        # make dataloader
        image_dataset_generator = dataset_generator(file_list, target_list, batch_size=self.parameter_config['batch_size'], phase=self.control_config["phase"], train_valid_split=self.parameter_config['train_valid_split'], valid_ratio=self.parameter_config['valid_ratio'], stratify=target_list, random_seed=self.parameter_config['split_seed'])
        dataloader = image_dataset_generator.dataloader()

        # build model
        vision_model = ModelFactory(model_name=self.parameter_config["vision_model_structure"],
                                   pretrained=self.parameter_config["pretrained"],
                                   class_num=len(uni_label))

        # get loss function from LossFactory
        loss_fn = LossFactory(loss_name=self.parameter_config["loss_name"]).get_loss_fn()

        # get optimizer from OptimizerFactory
        optimizer_kwargs =parse_kwargs(self.parameter_config,start_with="optimizer_")
        optimizer = OptimizerFactory(optimizer_name=self.parameter_config["optimizer_name"], model=vision_model,
                                    **optimizer_kwargs).get_optimizer_fn()

        # get trainer from trainer_factory
        trainer = SupervisedTraining(epoch=self.parameter_config["epochs"],
                          result_model_path=self.control_config['save_model_path'])

        # train
        trainer.train(vision_model, dataloader['train'], dataloader['valid'], loss_fn, optimizer, gpu=self.parameter_config['gpu_use'])


    def inference(self):
        """Vision Machine's inference function.
        Parameters should be defined with config path (json type)
        when class instantiated.
            
        Returns
        -----------
        float
            best_value with decided eval metric, 
        str
            result_model_path
        """

        # find file list
        file_list, uni_label, target_list = prepare_data(train_path = self.control_config['test_data_dir'],
                                        remove_filename_list=self.control_config["remove_filename_list"])
    

        # make dataloader
        test_image_dataset_generator = dataset_generator(file_list, target_list, batch_size=self.parameter_config['batch_size'], phase=self.control_config["phase"], train_valid_split=self.parameter_config['train_valid_split'], valid_ratio=self.parameter_config['valid_ratio'], stratify=target_list, random_seed=self.parameter_config['split_seed'])
        test_data_gen = test_image_dataset_generator.dataloader()['test']

        # Load trained model
        if os.path.exists(self.control_config["load_model_path"]):
            vision_model = ModelFactory(model_name=self.parameter_config["vision_model_structure"],
                                        pretrained=False,
                                        class_num=len(uni_label))
            vision_model.load_state_dict(torch.load(self.control_config["load_model_path"]))
            logging_utils.info(f"Loading a model is finished.")
        else:
            logging_utils.error("Please Provide Correct Model Path.")


        # Inference
        tester = Tester(model = vision_model, test_data_gen = test_data_gen)
        prediction = tester.inference(gpu = True)
        tester.make_csv_report(file_list, prediction, target_list, self.control_config['inference_result_path'])
        tester.plot_confusion_matrix(target_list, prediction, self.control_config['inference_result_path'])



    def _load_config(self,config):
        """Load Json type configuration and put the parameters on Memory 

        Parameters
        -----------
        cfg_path : str
            json type configuration path
        
        Raises
        -------
        ValueError
            when tried to use invalid JSON File
        """
            
        try:
            if isinstance(config, str):
                with open(config, 'r') as cfg:
                    self.config = json.load(cfg)
                self.control_config = self.config['controls']
                self.parameter_config = self.config['parameters']
            elif isinstance(config, dict):
                self.config = config
                self.control_config = self.config['controls']
                self.parameter_config = self.config['parameters']
            else:
                self.config = None
                logging_utils.initiate_log("config_failure.log")
                logging_utils.error("config only takes JSON-type file path or " + \
                    "dictionary type")
        except Exception as e:
            logging_utils.initiate_log("config_failure.log")
            logging_utils.error(e)


if __name__ == "__main__":
    vision_machine = Vision_Machine("./vision_machine_parameters.params")
    if vision_machine.control_config['phase'] == 'train':
        vision_machine.train()
    else:
        vision_machine.inference()
