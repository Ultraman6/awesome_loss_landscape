"""
A library of pre-written evaluation functions for PyTorch loss functions.

The classes and functions in this module cover common loss landscape evaluations. In particular,
computing the loss, the gradient of the loss (w.r.t. model parameters) and Hessian of the loss
(w.r.t. model parameters) for some supervised learning loss is easily accomplished.
"""
from abc import abstractmethod
import torch
import torch.autograd
from torch.utils.data import TensorDataset, DataLoader
from hess_vec_prod import min_max_hessian_eigs
from landscape.model_wrapper import ModelWrapper

def get_metric(metric, criterion, inputs, targets, **kwargs):
    if metric == 'loss':
        return Loss(fn=criterion, inputs=inputs, targets=targets, **kwargs)
    elif metric == 'eigen':
        return Hessian(fn=criterion, inputs=inputs, targets=targets, **kwargs)
    elif metric == 'logit':
        return Logit(fn=criterion, inputs=inputs, targets=targets, **kwargs)
    else:
        raise NotImplementedError(f"Unsupported metric: {metric}")

def validate_metric(metric):
    """验证metric函数有效性"""
    if not callable(metric):
        raise ValueError("metric must be a callable function")

class Metric:
    """ Abstract class for PyTorch supervised learning loss evaluation functions. """
    def __init__(self, fn, inputs: torch.Tensor, targets: torch.Tensor, batch_size=-1, dataloader=None):
        super().__init__()
        self.fn = fn
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.dataloader = dataloader
        if batch_size > 0 and not dataloader:
            self.dataloader = DataLoader(TensorDataset(self.inputs, self.targets),
                                         batch_size=self.batch_size)

    @abstractmethod
    def __call__(self, model_wrapper, **kwargs):
        pass

    def to(self, device):
        self.inputs = self.inputs.to(device)
        self.targets = self.targets.to(device)

    def __iter__(self):
        for inputs, targets in self.dataloader:
            yield inputs, targets

    def to_dict(self):
        return {'fn': self.fn, 'inputs': self.inputs, 'targets':
            self.targets, 'batch_size': self.batch_size, 'dataloader': self.dataloader}

class Loss(Metric):
    """ Computes a specified loss function over specified input-output pairs. """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, model_wrapper: ModelWrapper, inputs:torch.Tensor=None, targets:torch.Tensor=None) -> float:
        if inputs is not None and targets is not None:
            return self.fn(model_wrapper.forward(inputs), targets).item()
        elif self.dataloader is not None:
            device = model_wrapper.device
            loss, num = 0.0, 0
            for X, Y in self:
                X, Y = X.to(device), Y.to(device)
                loss += self.fn(model_wrapper.forward(X), Y).item()
                num += Y.size(0)
            return loss / num
        else:
            return self.fn(model_wrapper.forward(self.inputs), self.targets).item()

class Logit(Metric):
    """ Computes the gradient of a specified loss function w.r.t. the model parameters
    over specified input-output pairs. """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, model_wrapper: ModelWrapper, **kwargs):
        logit = self.fn(model_wrapper.forward(self.inputs))
        return logit

class Hessian(Metric):
    """
    Computes the Hessian of a specified loss function w.r.t. the model
    parameters over specified input-output pairs.
    """
    def __init__(self, top_n=1, **kwargs):
        super().__init__(**kwargs)
        self.top_n = top_n

    def __call__(self, model_wrapper, **kwargs) -> tuple:
        model_wrapper.requires_grad_(True)
        if not self.dataloader:
            self.dataloader = DataLoader(TensorDataset(self.inputs, self.targets),
                         batch_size=self.batch_size if self.batch_size > 0 else 128)
        eigvals, eigvecs = min_max_hessian_eigs(model_wrapper.module, self.dataloader,
                                                self.fn, device=model_wrapper.device, verbose=True)
        eigvecs = [eigvecs[:, i] for i in range(eigvecs.shape[1])]
        model_wrapper.requires_grad_(False)
        return eigvals, eigvecs
