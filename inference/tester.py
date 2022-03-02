import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


class Tester(object):
    """Tester for trained model

    Parameters
    ----------
    model : str
        Trained torch model.
    test_data_gen : bool
        Test image data generator.
    """
    def __init__(self, model, test_data_gen):
        # about training strategy
        self.model = model
        self.test_data_gen = test_data_gen

        
    def inference(self, gpu = True):
        """
        Conduct inference

        Parameters
        ----------
        gpu
            Whether use GPU. Default is 'True'.

        Returns
        --------
        predictions
            predictions per time slices
        """

        # prediction

        print('start prediction')
        predictions = []

        if gpu:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
        with torch.no_grad():  
            for data in self.test_data_gen:
                images, labels = data['image'].float().to(device), data['target'].float().to(device)
                images = images.to(device)  
                labels = labels.to(device)  
                self.model.eval()  
                yhat = self.model(images)  
                pred = yhat.argmax(dim=1, keepdim = False)
                pred = list(pred.cpu().numpy())
                predictions = predictions + pred

        return predictions


    def make_csv_report(self, file_name_list, prediction, file_label_list, csv_path):
        ground_truth = np.argmax(file_label_list, axis=-1)
        matching = prediction == ground_truth
        csv_df = pd.DataFrame({'File_name': file_name_list, 'True_class': ground_truth, 'Predicted_class': prediction, 'Matching': matching})
        csv_df.to_csv(csv_path + 'prediction.csv')

    def plot_confusion_matrix(self, file_label_list, prediction, cm_path):
        ground_truth = np.argmax(file_label_list, axis=-1)
        distribution = confusion_matrix(ground_truth, prediction)
        plt.figure()
        ax = sns.heatmap(distribution, annot=True)
        plt.title('Confusion_matrix')
        plt.savefig(cm_path + 'confusion_matrix.png')
        print(distribution)




