import importlib, torch, torchmetrics
from argparse import Namespace
import pytorch_lightning as pl
from torch.optim import SGD, Adagrad, Adam, RMSprop
import torch.nn.functional as F

def load_model(args):
    module = importlib.import_module('.'.join(['src', 'datasets', args.dataset]))
    net = module.models[args.model](args)
    net.eval()
    return net

def configure_criterion(criterion):
    if criterion == "ce":
        return F.cross_entropy
    elif criterion == "bce":
        return F.binary_cross_entropy
    elif criterion == "mse":
        return F.mse_loss
    else:
        raise Exception(f"Criterion not recognized: {criterion}")

class GenericModel(pl.LightningModule):
    """GenericModel class that enables flattening of the models parameters."""

    def __init__(self, args: Namespace):
        super().__init__()
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
        self.loss_fn = configure_criterion(args.criterion)
        self.gpu = args.gpu
        self.num_classes = args.num_classes
        self.optim_path = []
        self.acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=args.num_classes
        )

    def _step(self, batch):
        x, y = batch
        preds = self(x)
        if self.num_classes == 1:
            preds = preds.flatten()
            y = y.float()
        loss = self.loss_fn(preds, y)
        acc = self.acc(preds, y)
        return loss, acc

    def configure_optimizers(self):
        """Configure the optimizer for Pytorch Lightning.

        Raises:
            Exception: Optimizer not recognized.
        """
        if self.optimizer == "adam":
            return Adam(self.parameters(), self.lr)
        elif self.optimizer == "sgd":
            return SGD(self.parameters(), self.lr)
        elif self.optimizer == "adagrad":
            return Adagrad(self.parameters(), self.lr)
        elif self.optimizer == "rmsprop":
            return RMSprop(self.parameters(), self.lr)
        else:
            raise Exception(
                f"custom_optimizer supplied is not supported: {self.custom_optimizer}"
            )

    # filter param for flat analysis
    def get_flat_params(self):
        """Get flattened and concatenated params of the models."""
        params = self._get_params()
        flat_params = torch.Tensor()
        if torch.cuda.is_available() and self.gpu:
            flat_params = flat_params.cuda()
        elif torch.backends.mps.is_built():
            flat_params = flat_params.to("mps")
        else:
            flat_params = flat_params
        for _, param in params.items():
            flat_params = torch.cat((flat_params, torch.flatten(param)))
        return flat_params

    def init_from_flat_params(self, flat_params):
        """Set all models parameters from the flattened form."""
        if not isinstance(flat_params, torch.Tensor):
            raise AttributeError(
                "Argument to init_from_flat_params() must be torch.Tensor"
            )
        shapes = self._get_param_shapes()
        state_dict = self._unflatten_to_state_dict(flat_params, shapes)
        self.load_state_dict(state_dict, strict=True)

    def _get_param_shapes(self):
        shapes = []
        for name, param in self.named_parameters():
            shapes.append((name, param.shape, param.numel()))
        return shapes

    def _get_params(self):
        params = {}
        for name, param in self.named_parameters():
            if torch.cuda.is_available() and self.gpu:
                params[name] = param.data.cuda()
            elif torch.backends.mps.is_built():
                params[name] = param.data.to("mps")
            else:
                params[name] = param.data
        return params

    def _unflatten_to_state_dict(self, flat_w, shapes):
        state_dict = {}
        counter = 0
        for shape in shapes:
            name, tsize, tnum = shape
            param = flat_w[counter : counter + tnum].reshape(tsize)
            state_dict[name] = torch.nn.Parameter(param)
            counter += tnum
        assert counter == len(flat_w), "counter must reach the end of weight vector"
        return state_dict

    def loss_fn(self, y_pred, y):
        """Loss function."""

        return self.loss_fn(y_pred, y)

    def on_train_epoch_end(self):
        """Saves all steps in each epoch."""
        if int(self.current_epoch) == 150 or int(self.current_epoch) == 225 or int(self.current_epoch) == 275:
            self.lr *= self.lr_decay
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.lr

    # def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    #     print(checkpoint.keys())
    #     checkpoint['train_Loss']

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return {'loss': loss, 'acc': acc}
        return loss, acc

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {'loss': loss, 'acc': acc}


    def test_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return {'loss': loss, 'acc': acc}
        return loss, acc