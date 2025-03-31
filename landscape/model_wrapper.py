""" Class used to define interface to complex model """
import abc
import copy

import torch.nn
from torch import nn

from landscape.model_parameters import ModelParameters

class ModelWrapper(abc.ABC):
    def __init__(self, module: nn.Module, dtype='weight', ignore='biasbn'):
        self.module = module
        self.dtype = dtype
        self.ignore = ignore
        self.device = getattr(module, 'device', 'cpu')

    def get_module(self) -> nn.Module:
        return self.module

    def get_module_parameters(self) -> ModelParameters:
        param_list = []
        if self.dtype == 'weight':
            params = self.module.parameters()
        elif self.dtype == 'state':
            params = self.module.state_dict().values()
        else:
            raise NotImplementedError(f'Unknown dtype: {self.dtype}')
        for p in params:
            if p.dim() > 1 or self.ignore != 'biasbn':
                param_list.append(p)

        return ModelParameters(param_list)

    def train(self, mode=True) -> 'ModelWrapper':
        self.module.train(mode)
        return self

    def to(self, device):
        self.module = self.module.to(device)
        self.device = device
        return self

    def eval(self) -> 'ModelWrapper':
        return self.train(False)

    def requires_grad_(self, requires_grad=True) -> 'ModelWrapper':
        for p in self.module.parameters():
            p.requires_grad = requires_grad
        return self

    def zero_grad(self) -> 'ModelWrapper':
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        return self

    def parameters(self):
        return list(self.module.parameters())

    def named_parameters(self):
        return list(self.module.named_parameters())

    def get_grads(self):
        return  torch.cat([p.grad for p in self.module.parameters()], dim=0)

    @abc.abstractmethod
    def forward(self, x):
        pass

class SimpleModelWrapper(ModelWrapper):
    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__(model, **kwargs)

    def forward(self, x):
        return self.module(x)

class GeneralModelWrapper(ModelWrapper):
    def __init__(self, model, forward_fn):
        super().__init__(model)
        self.model = model
        self.forward_fn = forward_fn

    def forward(self, x):
        return self.forward_fn(self.model, x)

def wrap_model(model, deepcopy=True, require_grad=False, **kwargs):
    if deepcopy:
        model = copy.deepcopy(model)
    if isinstance(model, ModelWrapper):
        return model.requires_grad_(require_grad)
    elif isinstance(model, torch.nn.Module):
        return SimpleModelWrapper(model, **kwargs).requires_grad_(require_grad)
    else:
        raise ValueError('Only model of type torch.nn.modules.module.Module can be passed without a wrapper.')
