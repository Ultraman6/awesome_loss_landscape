import os, sys
import concurrent.futures

from benchmark.boundary import perturb

root = os.path.dirname(os.getcwd())
sys.path.append(root)
import torch.distributed as dist
import umap
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import graphlearning as gl
import copy
import pandas as pd
import re
import atexit
from argparse import Namespace
import importlib
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam
from tqdm import tqdm
from benchmark.metrics import LossMetric, AccuracyMetric, ProtoPCAMetric
from options import args
import numpy as np
import torch.multiprocessing as mp

def visual_prototype(features, labels, outputs=None, dirs=None, method:str = "pca", desc=None, path=None):
    print(f"reducing by {method} with features shape: {features.shape}, "
          f"labels shape: {labels.shape}, outputs shape: {outputs.shape}")
    if method == "pca":
        pca = PCA(n_components=2)
        reduce_features = pca.fit_transform(features)
    elif method == "tsne":
        tsne = TSNE(n_components=2, random_state=42, verbose=1)
        reduce_features = tsne.fit_transform(features)
    elif method == "umap":
        umap_reducer = umap.UMAP(verbose=True)
        reduce_features = umap_reducer.fit_transform(features)
    elif method == 'proj':
        if dirs is None:
            raise ValueError("dirs must be provided when method is 'eigen'.")
        print(dirs.shape)
        reduce_features = np.dot(features, dirs.T)
    elif method == 'ars':
        reduce_features = gl.graph.ars(features, prog=True)
    else:
        raise ValueError(f"Invalid method: {method}")

    # 创建一个 1 行 2 列的子图布局
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(reduce_features[:, 0], reduce_features[:, 1], c=labels, cmap='viridis')
    plt.title(f'{method.upper()} - Labels')
    if outputs is not None:
        plt.subplot(1, 2, 2)
        plt.scatter(reduce_features[:, 0], reduce_features[:, 1], c=outputs, cmap='viridis')
        plt.title(f'{method.upper()} - Outputs')
    if desc:
        plt.figtext(0.5, 0.01, desc, ha='center', va='bottom', fontsize=10)
    plt.show()
    if path:
        plt.savefig(f"{path}_{method}.png")
        print(f"图像已保存到 {path}_{method}.png。")

def print_distribution(datamodule):
    # 处理训练数据
    _, Y_train = datamodule.train_data()
    unique_train, counts_train = torch.unique(Y_train, return_counts=True)
    total_train = len(Y_train)
    distribution_train = counts_train.float() / total_train
    print("train", distribution_train.numpy())

    # 处理评估数据
    _, Y_eval = datamodule.eval_data()
    unique_eval, counts_eval = torch.unique(Y_eval, return_counts=True)
    total_eval = len(Y_eval)
    distribution_eval = counts_eval.float() / total_eval
    print("test", distribution_eval.numpy())

def load_data(args: Namespace):
    module = importlib.import_module('.'.join(['benchmark', 'dataset', args.dataset]))
    data = module.DataModule(args)
    if args.num_classes is None:
        args.num_classes = data.num_classes
    if not args.input_dim:
        args.input_dim = data.input_dim
    return data

def load_model(args: Namespace):
    module = importlib.import_module('.'.join(['benchmark', 'dataset', args.dataset]))
    net = module.models[args.model](args)
    net.eval()
    net.cpu()
    return net

def configure_criterion(args: Namespace):
    criterion = getattr(args, 'criterion', 'ce')
    if criterion == "ce":
        return F.cross_entropy
    elif criterion == "bce":
        return F.binary_cross_entropy
    elif criterion == "mse":
        return F.mse_loss
    else:
        raise Exception(f"Criterion not recognized: {criterion}")

def configure_scheduler(optimizer, args: Namespace):
    scheduler = getattr(args, 'scheduler', 'step')
    lr_decay = getattr(args, 'lr_decay', 0.1)
    lr_decay_step = getattr(args, 'lr_decay_step', 0)
    milestones = getattr(args, 'milestones', [])
    T_max = getattr(args, 'T_max', 0)
    eta_min = getattr(args, 'eta_min', 0)
    if scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, gamma=lr_decay, step_size=lr_decay_step)
    elif scheduler == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=lr_decay, milestones=milestones)
    elif scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else:
        raise ValueError('Unknown scheduler: {}'.format(scheduler))

def configure_optimizer(model, args: Namespace):
    """Configure the optimizer for Pytorch Lightning.

    Raises:
        Exception: Optimizer not recognized.
    """
    optimizer = getattr(args, 'optimizer', 'sgd')
    lr = getattr(args, 'lr', 0.01)
    weight_decay = getattr(args, 'weight_decay', 0)
    momentum = getattr(args, 'momentum', 0)
    betas = getattr(args, 'betas', (0.9, 0.999))
    eps = getattr(args, 'eps', 1e-8)

    if optimizer == "sgd":
        return SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == "adam":
        return Adam(model.parameters(), lr, weight_decay=weight_decay, betas=betas, eps=eps)
    else:
        raise Exception(
            f"custom_optimizer supplied is not supported: {optimizer}"
        )

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def proto_eigen(model, metric, X, Y):
    protos = model.encoder(X).detach()
    metric.update(model.head, Y, protos)
    preds = model(X).detach()
    return (Y.cpu().numpy(), protos.detach().cpu().numpy(),
            F.log_softmax(preds, dim=1).detach().cpu().numpy())

def proto(model, X, Y):
    protos = model.encoder(X).detach()
    preds = model(X).detach()
    return (Y.cpu().numpy(), protos.detach().cpu().numpy(),
            F.log_softmax(preds, dim=1).detach().cpu().numpy())

class Trainer:
    epoch = 0
    train_losses, train_accuracies = {}, {}
    eval_losses, eval_accuracies = {}, {}
    train_losses_by_cls, train_accuracies_by_cls = {}, {}
    eval_losses_by_cls, eval_accuracies_by_cls = {}, {}
    proto = {}
    def __init__(self, args: Namespace):
        self.args = args
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.device = args.device
        self.data = load_data(args)
        print_distribution(self.data)
        self.eval = getattr(args, 'eval', False)
        self.save = getattr(args, 'save', False)
        self.model = load_model(args)
        self.optimizer = configure_optimizer(self.model, args)
        self.criterion = configure_criterion(self.args)
        self.ckpt_root = os.path.join(args.save_root, 'checkpoints')
        self.log_root = os.path.join(args.save_root, 'logs')
        self.record_cls = getattr(args, 'record_cls', False)
        self.loss_metric = LossMetric(self.args.num_classes, self.criterion, self.record_cls)
        self.acc_metric = AccuracyMetric(self.args.num_classes, self.record_cls)
        if args.resume_epoch != 0:
            self.resume()
        self.dir = None

    def resume_filepath(self):
        if self.args.resume_epoch == -1:
            filename = 'last'
        elif self.args.resume_epoch == -2:
            filename = 'best'
        else:
            filename = f"epoch={self.args.resume_epoch}"
        return str(os.path.join(self.ckpt_root, f"{filename}.ckpt"))

    def resume(self):
        filepath = self.resume_filepath()
        store = torch.load(filepath, weights_only=False)
        self.model.load_state_dict(store['model_state_dict'])
        self.optimizer.load_state_dict(store['optimizer_state_dict'])
        self.epoch = store['epoch']
        self.lr = store['lr']

        print(f"Resume from {filepath}, epoch: {self.epoch}, lr: {self.lr}")

    def save_checkpoint(self, is_last=False, is_init=False):
        if is_init:
            filename = 'init.ckpt'
        elif is_last:
            filename = 'last.ckpt'
        else:
            filename = f"epoch={self.epoch}.ckpt"
        ckpt_path = os.path.join(self.ckpt_root, filename)
        store = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[self.epoch],
            'train_acc': self.train_accuracies[self.epoch],
            'epoch': self.epoch,
            'lr': self.lr
        }
        if self.eval:
            store.update({'eval_loss': self.eval_losses[self.epoch],
                          'eval_acc': self.eval_accuracies[self.epoch]})
        torch.save(store, ckpt_path)
        print(f"Checkpoint saved at {ckpt_path}.")

        if (not is_last and not is_init and
            self.eval_losses[self.epoch] == min(self.eval_losses.values())) \
                if self.eval else (
            self.train_losses[self.epoch] == min(self.train_losses.values())):
            store['epoch'] = self.epoch
            torch.save(store, os.path.join(self.ckpt_root, 'best.ckpt'))

    def load_checkpoint(self):
        ckpt_path = os.path.join(self.ckpt_root, 'checkpoints', f"epoch={self.args.resume_epoch}.ckpt")
        checkpoint = torch.load(ckpt_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = self.args.resume_epoch
        print(f"Checkpoint loaded, starting at {ckpt_path}.")

    def run(self):
        os.makedirs(self.ckpt_root, exist_ok=True)
        os.makedirs(self.log_root, exist_ok=True)

        self.model.to(self.device)
        self.loss_metric.to(self.device)
        self.acc_metric.to(self.device)
        self.on_the_begin()
        while self.epoch <= self.args.epochs:
            self.train_step()
            if self.epoch % self.args.interval_epoch == 0:
                if self.eval:
                    self.eval_step()
                if self.save:
                    self.save_checkpoint()
            self.update_lr()
            self.epoch += 1
        self.on_the_end()

    @torch.enable_grad()
    def train_step(self):
        pbar = tqdm(total=len(self.data.train_loader()),
                    desc=f"Training Epoch: {self.epoch}")
        self.model.train().to(self.device)
        self.loss_metric.reset()
        self.acc_metric.reset()
        for (inputs, targets) in self.data.train_loader():
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)

            loss = self.loss_metric(outputs, targets)
            acc = self.acc_metric(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            pbar.set_postfix({"batch_loss": round(loss.item(), 4), "batch_acc": round(acc.item(), 2)})
            pbar.update(1)

        self.log_metrics(pbar, train=True)

    @torch.no_grad()
    def eval_step(self, train:bool=False):
        dataloader = self.data.train_loader() if train else self.data.eval_loader()
        pbar = tqdm(total=len(dataloader),
                    desc=f"Testing Epoch: {self.epoch} on "
                         f"{'train' if train else 'test'}")

        self.model.eval()
        self.loss_metric.reset()
        self.acc_metric.reset()

        for (inputs, targets) in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_metric(outputs, targets)
            acc = self.acc_metric(outputs, targets)

            pbar.set_postfix({"batch_loss": round(loss.item(), 4), "batch_acc": round(acc.item(), 2)})
            pbar.update(1)

        self.log_metrics(pbar, train=train)

    @torch.no_grad()
    def proto_step(self, root:str, eval:bool=False, soft:bool=False, save:bool=True, cls_list:tuple=()):
        filepath = os.path.join(root,
                                f"embedding_epoch={self.epoch}_"
                                f"classes={str(cls_list)}_"
                                f"{'eval' if eval else 'train'}.ckpt")
        if os.path.exists(filepath):
            store = torch.load(filepath, weights_only=False)
        else:
            store = {}

        if ('embeddings' in store and 'labels' in store and
            f"{'soft' if soft else 'hard'}_logits" in store):
            print(f"load embeddings、labels、logits from {filepath}")
            return store.values()
        else:
            dataloader = self.data.eval_loader(cls_list) if eval else self.data.train_loader(cls_list)
            if not (hasattr(self.model, 'encoder') or hasattr(self.model, 'head')):
                raise ValueError(f"Encoder or Head not found in {self.model.__class__.__name__}.")
            self.model.eval().to(self.device)
            features, labels, outputs = [], [], []
            mp.set_start_method('spawn', force=True)
            with mp.Pool(processes=20) as pool:
                results = [pool.apply_async(proto, args=(self.model, X.to(self.device),
                                                         Y.to(self.device))) for X, Y in dataloader]
                for result in tqdm(results, desc=f"Proto Epoch: {self.epoch}"):
                    Y, preds, protos = result.get()
                    features.extend([p for p in protos])
                    labels.extend([y for y in Y])
                    outputs.extend([p if soft else np.argmax(p) for p in preds])

            features, labels, outputs = np.array(features), np.array(labels), np.array(outputs)
            if save:
                store = {'embeddings': features, 'labels': labels,
                         f"{'soft' if soft else 'hard'}_logits": outputs}
                torch.save(store, filepath)
                print(f"embeddings、labels、logits 、dirs、desc saved at {filepath}.")
            return features, labels, outputs

    def proto_eigen_step(self, root:str, eval:bool = False, soft:bool = False, top:int=2, order:int=0, save:bool=True, cls_list:tuple=()):
        filepath = os.path.join(root,
                                f"embedding_epoch={self.epoch}_"
                                f"classes={str(cls_list)}_"
                                f"{'eval' if eval else 'train'}.ckpt")
        if os.path.exists(filepath):
            store = torch.load(filepath, weights_only=False)
        else:
            store = {}
        if ('embeddings' in store and 'labels' in store and
            f"{'soft' if soft else 'hard'}_logits" in store and
            f"top={top}_order={order}_dirs" in store and
            f"top={top}_order={order}_desc" in store):
            print(f"load embeddings、labels、logits、dirs、desc from {filepath}")
            return store.values()
        else:
            data = self.data.eval_data(cls_list) if eval else self.data.train_data(cls_list)
            dataloader = [(x, y) for x, y in zip(*data)]
            if not (hasattr(self.model, 'encoder') or hasattr(self.model, 'head')):
                raise ValueError(f"Encoder or Head not found in {self.model.__class__.__name__}.")
            metric = ProtoPCAMetric(self.criterion, num_classes=self.args.num_classes, top=top, order=order)
            metric.to(self.device)
            self.model.eval().to(self.device)
            features, labels, outputs = [], [], []
            mp.set_start_method('spawn', force=True)
            with mp.Pool(processes=20) as pool:
                results = [pool.apply_async(proto_eigen, args=(self.model, metric, X.to(self.device).unsqueeze(0),
                                                               Y.to(self.device).unsqueeze(0))) for X, Y in dataloader]
                for result in tqdm(results, desc=f"Proto Eigen Epoch: {self.epoch}"):
                    Y, preds, protos = result.get()
                    features.extend([p for p in protos])
                    labels.extend([y for y in Y])
                    outputs.extend([p if soft else np.argmax(p) for p in preds])

            pca_vecs, pca_vars = metric._compute()
            features, labels, outputs = np.array(features), np.array(labels), np.array(outputs)
            desc = f"variance of each dim:{'_'.join(map(str, pca_vars))}"
            if save:
                store.update({'embeddings': features, 'labels': labels,
                              f"{'soft' if soft else 'hard'}_logits": outputs,
                              f"top={top}_order={order}_dirs": pca_vecs,
                              f"top={top}_order={order}_desc": desc})
                torch.save(store, filepath)
                print(f"embeddings、labels、logits 、dirs、desc saved at {filepath}.")
            return features, labels, outputs, pca_vecs, pca_vars

    def log_metrics(self, pbar:tqdm, train:bool=True):
        losses = self.train_losses if train else self.eval_losses
        accuracies = self.train_accuracies if train else self.eval_accuracies
        avg_loss = self.loss_metric._compute(multi_cls=False).item()
        avg_acc = self.acc_metric._compute(multi_cls=False).item()
        losses[self.epoch] = avg_loss
        accuracies[self.epoch] = avg_acc

        if self.record_cls:
            losses_cls = self.train_losses_by_cls if train else self.eval_losses_by_cls
            accuracies_cls = self.train_accuracies_by_cls if train else self.eval_accuracies_by_cls
            class_accuracies = self.acc_metric._compute(multi_cls=True)
            class_losses = self.loss_metric._compute(multi_cls=True)
            losses_by_cls = {i: class_losses[i].item() for i in range(self.args.num_classes)}
            accuracies_by_cls = {i: class_accuracies[i].item() for i in range(self.args.num_classes)}
            postfix_info = {
                "avg_loss": round(avg_loss, 4),
                "avg_acc": round(avg_acc, 2),
                "avg_cls_loss": [round(loss, 4) for loss in losses_by_cls.values()],
                "avg_cls_acc": [round(acc, 2) for acc in accuracies_by_cls.values()]
            }
            pbar.set_postfix(postfix_info)
            losses_cls[self.epoch] = losses_by_cls
            accuracies_cls[self.epoch] = accuracies_by_cls
        else:
            pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}", "avg_acc": f"{avg_acc:.2f}"})

    def update_lr(self):
        self.lr *= self.args.lr_decay ** (self.epoch - 1)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.args.lr_decay ** (self.epoch - 1)

    def on_the_begin(self):
        self.eval_step(train=True)
        if self.eval:
            self.eval_step()
        if self.save:
            self.save_checkpoint(is_init=True)
        self.epoch += 1
        self.update_lr()

    def on_the_end(self):
        self.epoch -= 1
        if self.save:
            self.save_checkpoint(is_last=True)
        headers = ['epoch', 'train_loss', 'train_acc', 'eval_loss', 'eval_acc']
        if self.record_cls:
            for i in range(self.args.num_classes):
                headers.extend([f'train_loss_cls_{i}', f'train_acc_cls_{i}', f'eval_loss_cls_{i}', f'eval_acc_cls_{i}'])

        df = pd.DataFrame(columns=headers)
        start_epoch = self.args.resume_epoch if self.args.resume_epoch > 0 else 1

        for epoch in range(start_epoch, self.args.epochs+1):
            row = {
                'epoch': epoch,
                'train_loss': self.train_losses[epoch],
                'train_acc': self.train_accuracies[epoch],
                'eval_loss': self.eval_losses[epoch] if epoch in self.eval_losses else None,
                'eval_acc': self.eval_accuracies[epoch] if epoch in self.eval_accuracies else None
            }
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        # 保存整体数据到 Excel 文件
        excel_path = os.path.join(self.log_root, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='overall', index=False)

            if self.record_cls:
                # 保存分类别的训练和评估数据到不同的 sheet
                class_metrics = {
                    'train_losses_by_cls': self.train_losses_by_cls,
                    'train_accuracies_by_cls': self.train_accuracies_by_cls,
                    'eval_losses_by_cls': self.eval_losses_by_cls,
                    'eval_accuracies_by_cls': self.eval_accuracies_by_cls
                }
                for metric_name, metric_data in class_metrics.items():
                    data = []
                    for epoch, metric in metric_data.items():
                        row = {'epoch': epoch}
                        for i in range(self.args.num_classes):
                            row[f'cls_{i}'] = metric[i]
                        data.append(row)
                    metric_df = pd.DataFrame(data)
                    metric_df.to_excel(writer, sheet_name=metric_name, index=False)

    def get_checkpoints(self, begin_epoch: int, end_epoch: int, center_epoch: int):
        if not hasattr(self, '_checkpoints'):
            pattern = r'epoch=(\d+)\.ckpt'
            result = {'model': None, 'epoch': None, 'loss': None, 'accuracy': None ,
                      'trajectory': [], 'epochs': [], 'losses': [], 'accuracies': []}
            def get_store(epoch:int):
                if epoch == -1:
                    filename = 'last.ckpt'
                elif epoch == -2:
                    filename = 'best.ckpt'
                elif epoch == 0:
                    filename = 'init.ckpt'
                else:
                    filename = f'epoch={epoch}.ckpt'
                return torch.load(os.path.join(self.ckpt_root, filename),
                                  weights_only=False)
            max_epoch_store = get_store(end_epoch)
            max_epoch = max_epoch_store['epoch']

            min_epoch_store = get_store(begin_epoch)
            min_epoch = min_epoch_store['epoch']
            model = copy.deepcopy(self.model)
            model.load_state_dict(min_epoch_store['model_state_dict'])
            result['trajectory'].append(model)
            result['epochs'].append(min_epoch)
            result['losses'].append(getattr(min_epoch_store, 'eval_loss',
                                            min_epoch_store['train_loss']))
            result['accuracies'].append(getattr(min_epoch_store, 'eval_acc',
                                                min_epoch_store['train_acc']))

            middle_epoch_store = get_store(center_epoch)
            middle_epoch = middle_epoch_store['epoch']
            model = copy.deepcopy(self.model)
            model.load_state_dict(middle_epoch_store['model_state_dict'])
            result['model'] = model
            result['epoch'] = middle_epoch
            result['loss'] = getattr(min_epoch_store, 'eval_loss',
                                            min_epoch_store['train_loss'])
            result['accuracy'] = getattr(min_epoch_store, 'eval_acc',
                                                min_epoch_store['train_acc'])

            for root, dirs, files in os.walk(self.ckpt_root):
                for file in files:
                    match = re.match(pattern, file)
                    if match:
                        epoch = int(match.group(1))
                        if epoch > min_epoch and epoch < max_epoch:
                            file_path = str(os.path.join(root, file))
                            try:
                                store = torch.load(file_path, weights_only=False)
                                model = copy.deepcopy(self.model)
                                model.load_state_dict(store['model_state_dict'])
                                result['epochs'].append(epoch)
                                result['trajectory'].append(model)
                                result['losses'].append(getattr(store, 'eval_loss', store['train_loss']))
                                result['accuracies'].append(getattr(store, 'eval_acc', store['train_acc']))
                            except Exception as e:
                                print(f"Error loading model from {file_path}: {e}")

            model = copy.deepcopy(self.model)
            model.load_state_dict(max_epoch_store['model_state_dict'])
            result['trajectory'].append(model)
            result['epochs'].append(max_epoch)
            result['losses'].append(getattr(max_epoch_store, 'eval_loss',
                                            max_epoch_store['train_loss']))
            result['accuracies'].append(getattr(max_epoch_store, 'eval_acc',
                                                max_epoch_store['train_acc']))
            if middle_epoch not in result['epochs']:
                middle_epoch_index = result['epochs'].index(middle_epoch)
                result['trajectory'].insert(middle_epoch_index, result['model'])
                result['epochs'].insert(middle_epoch_index, result['epoch'])
                result['losses'].insert(middle_epoch_index, result['loss'])
                result['accuracies'].insert(middle_epoch_index, result['accuracy'])
            self._checkpoints = result

        return self._checkpoints

def calculate_hessian(x, y, device, criterion, net):
    x, y = x.to(device).unsqueeze(0), y.to(device).unsqueeze(0)
    proto = net.encoder(x).detach()
    def loss_func(proto):
        return criterion(net.head(proto), y)
    hessian = torch.autograd.functional.hessian(loss_func, proto)
    torch.linalg.svd(hessian[0, :, 0, :])

def calculate_boundary(x, y, device, criterion, net):
    x, y = x.to(device).unsqueeze(0), y.to(device).unsqueeze(0)
    f = net.encoder(x).detach().requires_grad_(True)
    perturb(f, y, device, net.head, criterion)
    return hessian


if __name__ == '__main__':
    runner = Trainer(args)
    runner.model.to(runner.device)
    data = [(x, y) for x, y in zip(*runner.data.eval_data())]
    # x, y = data[0]
    # x = x.to(runner.device).unsqueeze(0)
    # y = y.to(runner.device).unsqueeze(0)
    # preds = runner.model(x)
    # runner.criterion(preds, y)
    # data = [(x, y) for x, y in zip(*runner.data.train_data())]
    mp.set_start_method('spawn', force=True)
    net = runner.model.to(runner.device).train()
    hessian_list = []
    with mp.Pool(processes=10) as pool:
        results = [pool.apply_async(calculate_hessian, args=(x, y, runner.device, runner.criterion, net)) for x, y in data]
        for result in tqdm(results):
            hessian_shape = result.get()


    @atexit.register
    def _():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("已清空 CUDA 显存。")
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            print("已清空 MPS 显存。")