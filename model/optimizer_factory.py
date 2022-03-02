from torch.optim import (Adam, RMSprop, SGD, Adamax)

AVAILABLE_OPTIMIZER = ['Adam', 'SGD', 'RMSProp','Adamax']

class OptimizerFactory(object):
    """A tool that construct a optimizer of model

    Parameters
    ----------
    optimizer_name : str
        name of optimizer. Defaults to 'Adam'.
    kwargs : list
        parameters of optimizer. (ex: lr, betas, eps ...)
    """

    def __init__(self, model, optimizer_name='Adam', **kwargs):

        self.model = model

        if len(kwargs) > 0 :            
            self.kwargs = kwargs

        if optimizer_name in AVAILABLE_OPTIMIZER:
            self._optimizer_name = optimizer_name
        else:    
            raise NotImplementedError('{} has not been implemented, use optimizer in {}'.format(self.optimizer_name,AVAILABLE_OPTIMIZER))
        
    def get_optimizer_fn(self):        
        """get pytorch optimizer function
        Returns
        -------        
        torch.optim        
            pytorch optimizer function        
        """        

        if self._optimizer_name == 'Adam':
            return Adam(params = self.model.parameters(),
                        lr=self.kwargs['learning_rate'],
                        betas = (self.kwargs['beta_1'], self.kwargs['beta_2']),
                        eps = self.kwargs['epsilon'],
                        weight_decay = self.kwargs['weight_decay'])
        elif self._optimizer_name == 'RMSProp':
            return RMSprop(params = self.model.parameters(),
                            lr=self.kwargs['learning_rate'],
                            alpha=self.kwargs['alpha'],
                            momentum=self.kwargs['momentum'],
                            epsilon=self.kwargs['epsilon'],
                            centered=self.kwargs['centered'],
                            weight_decay = self.kwargs['weight_decay'])
        elif self._optimizer_name == 'SGD':
            return SGD(params = self.model.parameters(),
                        lr=self.kwargs['learning_rate'], 
                        momentum=self.kwargs['momentum'], 
                        nesterov=self.kwargs['nesterov'],
                        weight_decay = self.kwargs['weight_decay'])
        elif self._optimizer_name == 'Adamax':
            return Adamax(params = self.model.parameters(),
                        lr=self.kwargs['learning_rate'],
                        betas = (self.kwargs['beta_1'], self.kwargs['beta_2']),
                        eps = self.kwargs['epsilon'],
                        weight_decay = self.kwargs['weight_decay'])
            