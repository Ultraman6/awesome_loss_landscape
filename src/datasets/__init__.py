"""PyTorch Lightning datamodules.
The dataset for the models to train on. To add your own, use the following as examples.
"""
# pylint: disable = no-member
import importlib

def load_data(args):
    module = importlib.import_module('.'.join(['src', 'datasets', args.dataset]))
    data = module.DataModule(args)
    args.num_classes = data.num_classes
    args.input_dim = data.input_dim
    return data
