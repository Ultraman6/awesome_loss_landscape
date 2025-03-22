import bisect
import copy
import csv
import re
import time
from argparse import Namespace
import importlib
import os
from datetime import datetime

import torch
import pickle
import torch.nn.functional as F
from sympy.sets.sets import set_function
from torch.optim import SGD, Adam
import torchmetrics
from tqdm import tqdm
from benchmarks import args

def load_data(args: Namespace):
    module = importlib.import_module('.'.join(['benchmarks', 'datasets', args.dataset]))
    data = module.DataModule(args)
    if args.num_classes is None:
        args.num_classes = data.num_classes
    if args.input_dim is None:
        args.input_dim = data.input_dim
    return data

def load_model(args: Namespace):
    module = importlib.import_module('.'.join(['benchmarks', 'datasets', args.dataset]))
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

class Runner:
    epoch=0
    train_losses, train_accuracies = [], []
    eval_losses, eval_accuracies = [], []
    def __init__(self, args: Namespace):
        self.args = args
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.lr_step = args.lr_step
        self.data = load_data(args)
        self.eval = getattr(args, 'eval', False)
        self.model = load_model(args)
        self.optimizer = configure_optimizer(self.model, args)
        self.criterion = configure_criterion(self.args)
        self.ckpt_root = os.path.join(args.save_root, 'checkpoints')
        self.log_root = os.path.join(args.save_root, 'logs')
        os.makedirs(self.ckpt_root, exist_ok=True)
        os.makedirs(self.log_root, exist_ok=True)
        self.acc = torchmetrics.Accuracy( # 类别数仅决定商量维度
            task="multiclass", num_classes=args.num_classes
        )
        if args.resume_epoch > 0:
            self.resume()

    def resume(self):
        ckpt_path = os.path.join(self.ckpt_root, f"epoch={self.args.resume_epoch}.ckpt")
        store = pickle.load(open(ckpt_path, 'rb'))
        self.model.load_state_dict(store['state_dict'])

    def save_checkpoint(self, is_last=False):
        filename = 'last' if is_last else f"epoch={self.epoch}.ckpt"
        ckpt_path = os.path.join(self.ckpt_root, filename)
        store = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1],
            'train_acc': self.train_accuracies[-1],
            'epoch': self.epoch,
            'lr': self.lr
        }
        if self.eval:
            store.update({'eval_loss': self.eval_losses[-1], 'eval_acc': self.eval_accuracies[-1]})
        torch.save(store, ckpt_path)
        print(f"Checkpoint saved at {ckpt_path}.")
        if (not is_last and
            self.eval_losses[-1] == max(self.eval_losses)) \
                if self.eval else (
            self.train_losses[-1] == max(self.train_losses)):
            store['epoch'] = self.epoch
            torch.save(store, os.path.join(self.ckpt_root, 'best.pt'))
        time.sleep(1)

    # 加载模型和优化器状态
    def load_checkpoint(self):
        ckpt_path = os.path.join(self.ckpt_root, 'checkpoints', f"epoch={self.args.resume_epoch}.ckpt")
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = self.args.resume_epoch
        print(f"Checkpoint loaded, starting at {ckpt_path}.")
        time.sleep(1)

    def run(self):
        self.model.to(self.args.device)
        self.acc.to(self.args.device)
        while self.epoch < self.args.epochs:
            self.train_step()
            if self.epoch % self.args.save_epoch == 0:
                if self.eval:
                    self.eval_step()
                self.save_checkpoint()
            self.update_lr()
            self.epoch += 1
        self.on_the_end()

    def train_step(self):
        pbar = tqdm(self.data.train_loader, total=self.data.train_vol, desc=f"Training Epoch: {self.epoch}")
        total_loss, total_acc = 0.0, 0.0
        self.model.train()
        for (inputs, targets) in self.data.train_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            acc = self.acc(outputs, targets).item()
            total_acc += acc
            total_loss += loss.item()
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "batch_acc": f"{acc:.2f}"})
            pbar.update(1)

        avg_loss, avg_acc = total_loss / len(pbar), total_acc / len(pbar)
        pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}", "avg_acc": f"{avg_acc:.2f}"})
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_acc)

    def eval_step(self):
        pbar = tqdm(total=self.data.eval_vol, desc=f"Testing Epoch: {self.epoch}")
        total_loss, total_acc = 0.0, 0.0
        self.model.eval()
        self.optimizer.zero_grad()
        for (inputs, targets) in self.data.eval_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            acc = self.acc(outputs, targets).item()
            total_acc += acc
            total_loss += loss.item()
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "batch_acc": f"{acc:.2f}"})
            pbar.update(1)

        avg_loss, avg_acc = total_loss / len(pbar), total_acc / len(pbar)
        pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}", "avg_acc": f"{avg_acc:.2f}"})
        self.eval_losses.append(avg_loss)
        self.eval_accuracies.append(avg_acc)

    def update_lr(self):
        flag = False
        if type(self.lr_step) == int and self.epoch % self.lr_step == 0:
            flag = True
        if type(self.lr_step) == tuple and self.epoch in self.lr_step:
            flag = True
        if flag:
            self.lr *= self.args.lr_decay
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.args.lr_decay

    def on_the_end(self):
        self.save_checkpoint(is_last=True)
        headers = ['epoch', 'train_loss', 'train_acc', 'eval_loss', 'eval_acc']
        with open(os.path.join(self.log_root, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'), 'w') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            idx = 0
            start_epoch = self.args.resume_epoch if self.args.resume_epoch > 0 else 0
            for i in range(start_epoch, self.args.epochs):
                row = {
                    'epoch': i,
                    'train_loss': self.train_losses[idx],
                    'train_acc': self.train_accuracies[idx],
                }
                if self.eval and i % self.args.save_epoch == 0:
                    row.update({
                        'eval_loss': self.eval_losses[idx],
                        'eval_acc': self.eval_accuracies[idx]
                    })
                idx += 1
                writer.writerow(row)

    def get_checkpoints(self, begin_epoch: int, end_epoch: int, center_epoch: int):
        pattern = r'epoch=(\d+)\.ckpt'
        result = {'model': None, 'epoch': None, 'trajectory': [], 'epochs': [], 'losses': [], 'accuracies': []}
        for root, dirs, files in os.walk(self.ckpt_root):
            for file in files:
                match = re.match(pattern, file)
                if match and int(match.group(1)) >= begin_epoch and int(match.group(1)) <= end_epoch:
                    epoch = int(match.group(1))
                    file_path = str(os.path.join(root, file))
                    try:
                        store = torch.load(file_path)
                        model = copy.deepcopy(self.model)
                        model.load_state_dict(store['model_state_dict'])
                        result['epochs'].append(epoch)
                        result['trajectory'].append(model)
                        result['losses'].append(getattr(store, 'eval_loss', store['train_loss']))
                        result['accuracies'].append(getattr(store, 'eval_acc', store['train_acc']))
                    except Exception as e:
                        print(f"Error loading model from {file_path}: {e}")
            if center_epoch < 0:
                if center_epoch == -1:
                    store = torch.load(os.path.join(self.ckpt_root, 'last.pt'))
                elif center_epoch == -2:
                    store = torch.load(os.path.join(self.ckpt_root, 'best.pt'))
                else:
                    raise ValueError(f"Invalid center_epoch: {center_epoch}")
                result['model'] = store['model']
                result['epoch'] = store['epoch']
                if store['epoch'] not in result['epochs']:
                    idx = bisect.bisect_left(result['epochs'], store['epoch'])
                    result['trajectory'].insert(idx, store['model'])
                    result['epochs'].insert(idx, store['epoch'])
                    result['losses'].insert(idx, store['loss'])
                    result['accuracies'].insert(idx, store['acc'])
            else:
                result['model'] = result['trajectory'][-1]
                result['epoch'] = result['epochs'][-1]
        return result

if __name__ == '__main__':
    runner = Runner(args)
    runner.run()