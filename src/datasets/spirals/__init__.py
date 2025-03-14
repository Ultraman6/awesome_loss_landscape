import pickle
import os
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.datasets.spirals.model import mlp

models = {
    'mlp'                   : mlp.MLP,
}

class SpiralsDataModule(pl.LightningDataModule):
    """Datamodule for the Spirals dataset."""

    def __init__(self, args):
        """Init a SpiralsDataModule object.

        Args:
            batch_size (optional): The batch size for training. Defaults to None meaning
              a batch_size of 1.
        """
        data_path = os.path.join(args.data_root, "SPIRALS", 'spirals.p')
        self.datadict = pickle.load(open(data_path, "rb"))
        self.input_dim = self.datadict["X_train"].shape[1]
        self.num_classes = len(set(self.datadict["y_train"]))
        self.X = torch.Tensor(self.datadict["X_train"]) # 全部 X 序列
        self.y = torch.LongTensor(self.datadict["y_train"]) # 全部 Y 序列
        self.dataset = TensorDataset(self.X, self.y)
        self.batch_size = args.batch_size if args.batch_size else len(self.X)

    def train_dataloader(self, num_workers=7):
        """Return the train dataloader for PyTorch Lightning.
        # 训练集以 bs 划分
        Args:
            num_workers (optional): Defaults to 0.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            persistent_workers=True,
        )