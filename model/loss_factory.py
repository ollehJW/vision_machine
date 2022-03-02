from torch.nn import (L1Loss, MSELoss, CrossEntropyLoss, NLLLoss, BCELoss)


AVAILABLE_LOSS = ['binary_crossentropy', 'crossentropy','MSE','MAE', 'negative-log-likelihood']

class LossFactory(object):
    """A tool that construct a loss of model

    Parameters
    ----------
    loss_name : str
        name of loss. Defaults to 'binary_crossentropy'.
  
    """

    def __init__(self, loss_name='binary_crossentropy'):
        if loss_name in AVAILABLE_LOSS:
            self.loss_name = loss_name
        else:    
            raise NotImplementedError('{} has not been implemented, use loss in {}'.format(self.loss_name,AVAILABLE_LOSS))
        
    def get_loss_fn(self):        
        """get pytorch loss function
        Returns
        -------        
        torch.nn.losses            
            pytorch loss function        
        """        
        loss_dict = {'binary_crossentropy': BCELoss(), 'crossentropy': CrossEntropyLoss(), 'MSE': MSELoss(), 'MAE':L1Loss(), 'negative-log-likelihood': NLLLoss()}
        return loss_dict.get(self.loss_name)