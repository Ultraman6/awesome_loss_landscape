import numpy as np
import torch
import torchmetrics
from torch.autograd.functional import hessian as Hessian
from sklearn.decomposition import PCA
from tqdm import tqdm
import os, sys
root = os.path.dirname(os.getcwd())
sys.path.append(root)

def append_reduce_fx(tensor_list):
    """
    自定义归约函数，将不同进程的张量列表合并成一个大列表
    """
    gathered = gather_all_tensors(tensor_list)
    combined_list = []
    for sub_list in gathered:
        combined_list.extend(sub_list)
    return combined_list

@torch.amp.autocast(device_type='cuda')
def hessian_metric(loss, parameters, desc: str='computing hessian by row', top: int = 1):
    grads = torch.autograd.grad(loss, parameters, create_graph=True, allow_unused=True)
    hessian = []
    grads_flat = torch.cat([g.flatten() for g in grads if g is not None])
    for g in tqdm(grads_flat, desc=desc, unit="row"):
        hessian_row = torch.autograd.grad(g, parameters, retain_graph=True, allow_unused=True)
        hessian_row = torch.cat([h.flatten() for h in hessian_row if h is not None])
        hessian.append(hessian_row)
    hessian = torch.stack(hessian).detach()
    U, S, _ = torch.linalg.svd(hessian)
    if top == -1:
        top_indices = [torch.arange(S.shape[1]) for _ in range(S.shape[0])]
    else:
        top_indices = torch.argsort(S, dim=1, descending=True)[:, :top]
    eigen_values = S[top_indices]
    eigen_vectors = U[:, top_indices]
    # 计算 Hessian 矩阵的迹
    hessian_trace = torch.trace(hessian)
    # 计算 Hessian 矩阵的密度
    hessian_density = torch.count_nonzero(hessian).item() / (hessian.numel())

    # 将返回值转换为普通类型
    eigen_values = eigen_values.cpu().numpy().tolist()
    eigen_vectors = eigen_vectors.cpu().numpy()
    hessian_trace = hessian_trace.cpu().item()
    hessian_density = float(hessian_density)
    return eigen_values, eigen_vectors, hessian_trace, hessian_density

class AccuracyMetric(torchmetrics.Metric):
    def __init__(self, num_classes, multi_cls=False, dist_sync_on_step=True):
        """
        初始化 AccuracyMetric 类
        :param num_classes: 类别数量
        :param multi_cls: 是否进行多类别准确率计算，默认 False
        :param dist_sync_on_step: 分布式同步于update or compute之后，默认 True
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.multi_cls = multi_cls
        # 注册用于存储批量总正确预测数量和样本总数的状态
        self.add_state("total_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("batch_sample", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("batch_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        # 注册用于存储每个类别正确预测数量和样本数量的状态
        if multi_cls:
            self.add_state("correct_per_class", default=torch.zeros(num_classes), dist_reduce_fx="sum")
            self.add_state("total_per_class", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds, targets):
        # 计算每个样本是否预测正确
        correct = (preds.argmax(dim=1) == targets)
        self.batch_correct = correct.sum()
        self.batch_sample = targets.size(0)
        self.total_correct += self.batch_correct
        self.total_samples += self.batch_sample

        if self.multi_cls:
            class_indices = torch.arange(self.num_classes, device=self.device)
            class_masks = (targets.unsqueeze(1) == class_indices).float()
            class_correct = (correct.unsqueeze(1) * class_masks).sum(dim=0)
            class_total = class_masks.sum(dim=0)
            self.correct_per_class += class_correct
            self.total_per_class += class_total

    def compute(self):
        pass

    def _compute(self, multi_cls:bool=False):
        """
        计算每个类别的准确率和批量平均准确率
        :return: 包含每个类别准确率的张量和批量平均准确率，或者仅返回批量平均准确率
        """
        if multi_cls and self.multi_cls:
            non_zero_mask = (self.total_per_class > 0)
            class_accuracies = torch.zeros(self.num_classes, dtype=torch.float32, device=self.device)
            class_accuracies[non_zero_mask] = (
                    self.correct_per_class[non_zero_mask] / self.total_per_class[non_zero_mask]).float()
            return class_accuracies
        else:
            if self.total_samples > 0:
                return self.total_correct / self.total_samples
            else:
                return torch.tensor(0.0, device=self.device)

    def reset(self):
        """
        重置所有类别的正确预测数量、样本数量、批量总正确预测数量和样本总数
        """
        self.total_correct.zero_()
        self.total_samples.zero_()
        if self.multi_cls:
            self.correct_per_class.zero_()
            self.total_per_class.zero_()

    def __call__(self, preds, targets):
        self.update(preds, targets)
        return self.batch_correct / self.batch_sample if self.batch_sample > 0 \
            else torch.tensor(0.0, device=self.device)

class LossMetric(torchmetrics.Metric):
    def __init__(self, num_classes, criterion, multi_cls=False, dist_sync_on_step=True):
        """
        初始化 LossMetric 类
        :param num_classes: 类别数量
        :param criterion: 外部传入的损失函数
        :param dist_sync_on_step: 是否在每个步骤同步分布式数据，默认 False
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.criterion = criterion
        self.multi_cls = multi_cls
        # 注册用于存储 batch 总损失和样本总数的状态
        self.add_state("total_batch_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_batch_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("batch_sample", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("batch_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        # 注册用于存储每个类别累计损失和样本数量的状态
        if multi_cls:
            self.add_state("total_loss_per_class", default=torch.zeros(num_classes), dist_reduce_fx="sum")
            self.add_state("total_samples_per_class", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    @torch.enable_grad()
    def update(self, preds, targets):
        """
        更新每个类别的损失值和 batch 总损失
        :param preds: 模型预测值
        :param targets: 真实标签
        """
        batch_losses = self.criterion(preds, targets, reduction='none')
        self.batch_sample = preds.size(0)
        self.batch_loss = batch_losses.sum()  # 千万别随便item()，直接销毁引用的计算图
        self.total_batch_loss += self.batch_loss.detach()
        self.total_batch_samples += self.batch_sample
        if self.multi_cls:
            class_indices = torch.arange(self.num_classes, device=self.device)
            class_masks = (targets.unsqueeze(1) == class_indices).float()
            class_losses = (batch_losses.detach().unsqueeze(1) * class_masks).sum(dim=0)
            class_samples = class_masks.sum(dim=0)
            self.total_loss_per_class += class_losses
            self.total_samples_per_class += class_samples

    def compute(self):
        pass

    def _compute(self, multi_cls:bool=False):
        """
        计算每个类别的平均损失和平均 batch 损失
        :return: 包含每个类别平均损失的张量和平均 batch 损失
        """
        # 避免除零错误，当某个类别样本数为 0 时，将其平均损失设为 0
        if multi_cls and self.multi_cls:
            non_zero_mask = self.total_samples_per_class > 0
            class_losses = torch.zeros(self.num_classes, dtype=torch.float32, device=self.device)
            class_losses[non_zero_mask] = self.total_loss_per_class[non_zero_mask] / self.total_samples_per_class[
                non_zero_mask]
            return class_losses
        else:
            if self.total_batch_samples > 0:
                return  self.total_batch_loss / self.total_batch_samples
            else:
                return torch.tensor(0.0, device=self.device)

    def reset(self):
        """
        重置所有类别的累计损失、样本数量、batch 总损失和样本总数
        """
        self.total_batch_loss.zero_()
        self.total_batch_samples.zero_()
        if self.multi_cls:
            self.total_loss_per_class.zero_()
            self.total_samples_per_class.zero_()

    def __call__(self, preds, targets):
        self.update(preds, targets)
        return self.batch_loss / self.batch_sample

class HessianMetric(torchmetrics.Metric):
    def __init__(self, criterion, parameters, num_classes, top=0, multi_cls=False, dist_sync_on_step=True):
        """
        初始化 HessianMetric 类
        :param num_classes: 类别数量
        :param criterion: 外部传入的损失函数
        :param multi_cls: 是否进行多类别计算，默认 False
        :param dist_sync_on_step: 是否在每个步骤同步分布式数据，默认 True
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.criterion = criterion
        self.multi_cls = multi_cls
        self.parameters = parameters
        self.top = top
        param_count = sum(p.numel() for p in parameters)
        self.add_state("hessian_sum", default=torch.zeros(param_count, param_count), dist_reduce_fx="sum")
        self.add_state("sample_num", default=torch.tensor(0), dist_reduce_fx="sum")
        if multi_cls:
            self.add_state("hessian_sum_per_cls", default=torch.zeros(num_classes, param_count, param_count),
                           dist_reduce_fx="sum")
            self.add_state("sample_num_per_cls", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    @torch.enable_grad()
    def update(self, preds, targets):
        """
        更新 Hessian 矩阵的相关信息
        :param preds: 模型预测值
        :param targets: 真实标签
        """
        batch_losses = self.criterion(preds, targets, reduction='none')
        for batch_loss, cls_idx in zip(batch_losses, targets):
            # 设置 allow_unused=True
            grads = torch.autograd.grad(batch_loss, self.parameters, create_graph=True, allow_unused=True)
            hessian = []
            for g in torch.cat([g.flatten() for g in grads if g is not None]):
                # 设置 allow_unused=True
                hessian_row = torch.autograd.grad(g, self.parameters, retain_graph=True, allow_unused=True)
                hessian_row = torch.cat([h.flatten() for h in hessian_row if h is not None])
                hessian.append(hessian_row)
            hessian = torch.stack(hessian).detach()
            self.hessian_sum += hessian
            self.sample_num += 1
            if self.multi_cls:
                cls_idx = int(cls_idx.item())
                self.hessian_sum_per_cls[cls_idx] += hessian
                self.sample_num_per_cls[cls_idx] += 1

    def _compute(self, multi_cls:bool=False):
        """
        计算并返回 Hessian 矩阵的奇异值和向量
        :return: 奇异值和向量
        """
        if self.multi_cls and multi_cls:
            mask = self.sample_num_per_class > 0
            hessian_per_class = self.hessian_per_class.clone()
            hessian_per_class[mask] /= self.sample_num_per_class[mask].unsqueeze(1).unsqueeze(2)
            U, S, _ = torch.linalg.svd(hessian_per_class[mask])
            if self.top == -1:
                top_indices = [torch.arange(S.shape[1]) for _ in range(S.shape[0])]
            else:
                top_indices = torch.argsort(S, dim=1, descending=True)[:, :self.top]
            eigen_values_per_cls = {cls_idx: S[i, top_indices[i]] for i, cls_idx in
                                    enumerate(torch.where(mask)[0])}
            eigen_vectors_per_cls = {cls_idx: U[i, :, top_indices[i]] for i, cls_idx in
                                     enumerate(torch.where(mask)[0])}
            return eigen_values_per_cls, eigen_vectors_per_cls
        else:
            hessian = self.hessian / self.sample_num
            U, S, _ = torch.linalg.svd(hessian)
            if self.top == -1:
                top_indices = [torch.arange(S.shape[1]) for _ in range(S.shape[0])]
            else:
                top_indices = torch.argsort(S, dim=1, descending=True)[:, :self.top]
            eigen_values = S[top_indices]
            eigen_vectors = U[:, top_indices]
            return eigen_values, eigen_vectors

    def reset(self):
        """
        重置相关状态
        """
        self.hessian_sum.zero_()
        self.sample_num.zero_()
        if self.multi_cls:
            self.hessian_sum_per_cls.zero_()
            self.sample_num_per_cls.zero_()

    def __call__(self, preds, targets, model_params):
        self.update(preds, targets, model_params)

    def compute(self):
        pass

# 计算特征原型的主成分方向，可设置0-2阶 item: 原型向量、梯度向量、奇异向量
class ProtoPCAMetric(torchmetrics.Metric):
    def __init__(self, criterion, num_classes, top=3, order=0, multi_cls=False, dist_sync_on_step=True):
        """
        初始化 ProtoMetric 类
        :param num_classes: 类别数量
        :param criterion: 外部传入的损失函数
        :param multi_cls: 是否进行多类别计算，默认 False
        :param dist_sync_on_step: 是否在每个步骤同步分布式数据，默认 True
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.criterion = criterion
        self.multi_cls = multi_cls
        self.top = top
        self.order = order
        self.add_state("item_list", default=[], dist_reduce_fx=append_reduce_fx)
        if multi_cls:
            self.add_state("item_list_per_cls", default=[[] for _ in range(num_classes)],
                           dist_reduce_fx=append_reduce_fx)

    @torch.enable_grad()
    def update(self, head, targets, protos):
        """
        更新 Hessian 矩阵的相关信息
        :param preds: 模型预测值
        :param targets: 真实标签
        """
        # 对每个样本计算损失，此时不需要 reduction='none'
        if self.order == 0:
            results = [p.detach().cpu().numpy() for p in protos]
        else:
            if self.order == 1:
                loss = self.criterion(head(protos), targets)
                grads = torch.autograd.grad(loss, protos, allow_unused=True)
                results = [g.detach().cpu().numpy() for g in grads]
            elif self.order == 2:
                results = []
                hessian = (Hessian(lambda protos: self.criterion(head(protos), targets), protos))
                for idx in range(targets.shape[0]):
                    U, _, _ = torch.linalg.svd(hessian[idx, :, idx, :])
                    results.append(U[0, :].detach().cpu().numpy())
            else:
                raise ValueError(f"Invalid order value: {self.order}. Order should be 0, 1, or 2.")

        self.item_list.extend(results)
        if self.multi_cls:
            for r, y in zip(results, targets):
                self.item_list_per_cls[y].append(r)

    def _compute(self, multi_cls:bool=False):
        """
        计算并返回 Hessian 矩阵的奇异值和向量
        :return: 奇异值和向量
        """
        if self.multi_cls and multi_cls:
            results = {}
            for cls in range(self.num_classes):
                array = self.item_list_per_cls[cls]
                if array:
                    pca = PCA(n_components=2)
                    pca.fit(np.array(array))
                    results[cls] = (pca.components_, pca.explained_variance_.tolist())
                else:
                    results[cls] = None
        else:
            pca = PCA(n_components=self.top)
            pca.fit(np.array(self.item_list))
            return pca.components_, pca.explained_variance_.tolist()

    def reset(self):
        """
        重置相关状态
        """
        self.item_list.clear()
        if self.multi_cls:
            self.item_list_by_cls.clear()

    def __call__(self, preds, targets, model_params):
        self.update(preds, targets, model_params)

    def compute(self):
        pass

if __name__ == '__main__':
    pass

