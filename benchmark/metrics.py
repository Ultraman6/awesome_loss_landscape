import argparse
import torch
import torchmetrics
from tqdm import tqdm
import os, sys
root = os.path.dirname(os.getcwd())
sys.path.append(root)

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

# class ProtoHessianMetric(torchmetrics.Metric):
#     def __init__(self, criterion, dim, num_classes, top=1, multi_cls=False, dist_sync_on_step=True):
#         """
#         :param num_classes: 类别数量
#         :param criterion: 外部传入的损失函数
#         :param dim: 隐藏特征的展平维度
#         :param multi_cls: 是否进行多类别计算，默认 False
#         :param dist_sync_on_step: 是否在每个步骤同步分布式数据，默认 True
#         """
#         super().__init__(dist_sync_on_step=dist_sync_on_step)
#         self.num_classes = num_classes
#         self.criterion = criterion
#         self.multi_cls = multi_cls
#         self.top = top
#         self.add_state("loss_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
#         self.add_state("proto_sum", default=torch.zeros(dim), dist_reduce_fx="sum")
#         self.add_state("sample_num", default=torch.tensor(0), dist_reduce_fx="sum")
#         if multi_cls:
#             self.add_state("loss_sum_per_cls", default=torch.zeros(num_classes, dim), dist_reduce_fx="sum")
#             self.add_state("sample_num_per_cls", default=torch.zeros(num_classes), dist_reduce_fx="sum")
#             self.add_state("proto_sum_per_cls", default=torch.tensor(0.0), dist_reduce_fx="sum")
#
#     @torch.enable_grad()
#     def update(self, preds, targets, protos):
#         """
#         更新损失的相关信息
#         :param preds: 模型预测值
#         :param targets: 真实标签
#         :param protos: 隐藏特征
#         """
#         batch_losses = self.criterion(preds, targets, reduction='none')
#         self.loss_sum += batch_losses.sum()
#         self.sample_num += len(targets)
#         if self.multi_cls:
#             for batch_loss, cls_idx, proto in zip(batch_losses, targets, protos):
#                 cls_idx = int(cls_idx.item())
#                 self.loss_sum_per_cls[cls_idx] += batch_loss
#                 self.proto_sum_per_cls[cls_idx] += proto.detech()
#                 self.sample_num_per_cls[cls_idx] += 1
#
#     @torch.enable_grad()
#     def _compute(self):
#         """
#         计算并返回 Hessian 矩阵的奇异值和向量
#         :return: 奇异值和向量
#         """
#         if self.multi_cls:
#             return self._compute_by_cls()
#         else:
#             return self._compute_all()
#
#     def _compute_all(self):
#         avg_loss = self.loss_sum / self.sample_num
#         avg_proto = self.proto_sum / self.sample_num
#         return hessian_metric(avg_loss, avg_proto, top=self.top)
#
#     def _compute_by_cls(self):
#         eigen_values_per_cls = {}
#         eigen_vectors_per_cls = {}
#         hessian_trace_per_cls = {}
#         hessian_density_per_cls = {}
#         for cls_idx in range(self.num_classes):
#             if self.sample_num_per_cls[cls_idx] > 0:
#                 avg_loss = self.loss_sum_per_cls[cls_idx] / self.sample_num_per_cls[cls_idx]
#                 avg_proto = self.proto_sum_per_cls[cls_idx] / self.sample_num_per_cls[cls_idx]
#                 (eigen_values_per_cls[cls_idx], eigen_vectors_per_cls[cls_idx],
#                  hessian_trace_per_cls[cls_idx], hessian_density_per_cls[cls_idx]) \
#                     = hessian_metric(avg_loss, avg_proto, top=self.top,
#                                desc=f'computing hessian for class{cls_idx} by row')
#         return eigen_values_per_cls, eigen_vectors_per_cls, hessian_trace_per_cls, hessian_density_per_cls
#
#     def reset(self):
#         """
#         重置相关状态
#         """
#         self.loss_sum.zero_()
#         self.sample_num.zero_()
#         if self.multi_cls:
#             self.loss_sum_per_cls.zero_()
#             self.sample_num_per_cls.zero_()
#
#     def __call__(self, preds, targets):
#         self.update(preds, targets)
#
#     def compute(self):
#         pass

class ProtoHessianMetric(torchmetrics.Metric):
    def __init__(self, criterion, dim, num_classes, top=0, multi_cls=False, dist_sync_on_step=True):
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
        self.top = top
        self.add_state("hessian_sum", default=torch.zeros(dim, dim), dist_reduce_fx="sum")
        self.add_state("sample_num", default=torch.tensor(0), dist_reduce_fx="sum")
        # if multi_cls:
        #     self.add_state("hessian_sum_per_cls", default=torch.zeros(num_classes, dim, dim),
        #                    dist_reduce_fx="sum")
        #     self.add_state("sample_num_per_cls", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    @torch.enable_grad()
    def update(self, preds, targets, protos):
        """
        更新 Hessian 矩阵的相关信息
        :param preds: 模型预测值
        :param targets: 真实标签
        """
        # 对每个样本计算损失，此时不需要 reduction='none'
        sample_loss = self.criterion(preds, targets)
        # 设置 allow_unused=True
        grads = torch.autograd.grad(sample_loss, protos, create_graph=True, allow_unused=True)
        hessian = []
        for g in torch.cat([g.flatten() for g in grads if g is not None]):
            # 设置 allow_unused=True
            hessian_row = torch.autograd.grad(g, protos, retain_graph=True, allow_unused=True)
            hessian_row = torch.cat([h.flatten() for h in hessian_row if h is not None])
            hessian.append(hessian_row)
        hessian = torch.stack(hessian).detach()
        self.hessian_sum += hessian
        self.sample_num += 1
        # if self.multi_cls:
        #     cls_idx = int(targets.item())
        #     self.hessian_sum_per_cls[cls_idx] += hessian
        #     self.sample_num_per_cls[cls_idx] += 1

    def _compute(self, multi_cls:bool=False):
        """
        计算并返回 Hessian 矩阵的奇异值和向量
        :return: 奇异值和向量
        """
        # if self.multi_cls and multi_cls:
        #     mask = self.sample_num_per_class > 0
        #     hessian_per_class = self.hessian_per_class.clone()
        #     hessian_per_class[mask] /= self.sample_num_per_class[mask].unsqueeze(1).unsqueeze(2)
        #     U, S, _ = torch.linalg.svd(hessian_per_class[mask])
        #     if self.top == -1:
        #         top_indices = [torch.arange(S.shape[1]) for _ in range(S.shape[0])]
        #     else:
        #         top_indices = torch.argsort(S, dim=1, descending=True)[:, :self.top]
        #     eigen_values_per_cls = {cls_idx: S[i, top_indices[i]] for i, cls_idx in
        #                             enumerate(torch.where(mask)[0])}
        #     eigen_vectors_per_cls = {cls_idx: U[i, :, top_indices[i]] for i, cls_idx in
        #                              enumerate(torch.where(mask)[0])}
        #     # 计算 Hessian 矩阵的迹
        #     return eigen_values_per_cls, eigen_vectors_per_cls
        # else:
        hessian = self.hessian / self.sample_num
        U, S, _ = torch.linalg.svd(hessian)
        if self.top == -1:
            top_indices = [torch.arange(S.shape[1]) for _ in range(S.shape[0])]
        else:
            top_indices = torch.argsort(S, dim=1, descending=True)[:, :self.top]
        eigen_values = S[top_indices]
        eigen_vectors = U[:, top_indices]
        hessian_trace = torch.trace(hessian)
        hessian_density = torch.count_nonzero(hessian).item() / (hessian.numel())
        return eigen_values, eigen_vectors, hessian_trace, hessian_density

    def reset(self):
        """
        重置相关状态
        """
        self.hessian_sum.zero_()
        self.sample_num.zero_()
        # if self.multi_cls:
        #     self.hessian_sum_per_cls.zero_()
        #     self.sample_num_per_cls.zero_()

    def __call__(self, preds, targets, model_params):
        self.update(preds, targets, model_params)

    def compute(self):
        pass

if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt

    # 检查 CUDA 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 更复杂的模型
    class ComplexModel(nn.Module):
        def __init__(self):
            super(ComplexModel, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(kernel_size=2)
            self.linear1 = nn.Linear(16 * 50, 64)
            self.linear2 = nn.Linear(64, 1)

        def forward(self, x):
            x = x.unsqueeze(1)  # 添加通道维度以适应卷积层输入
            x = self.conv1(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = x.view(-1, 16 * 50)
            x = self.linear1(x)
            x = torch.relu(x)
            x = self.linear2(x)
            return x

    # 生成假数据，调整输入数据长度为 100
    x = torch.randn(100, 100).to(device)
    y = 2 * x.mean(dim=1, keepdim=True) + 1 + 0.1 * torch.randn(100, 1).to(device)

    # 初始化模型、损失函数和优化器
    model = ComplexModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 方式一：每批更新时计算 Hessian
    hessian_sum_1 = None
    sample_num_1 = 0
    for _ in range(10):
        outputs = model(x)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward(create_graph=True)

        # 获取模型参数
        params = list(model.parameters())
        param_count = sum(p.numel() for p in params)
        hessian = torch.zeros(param_count, param_count).to(device)

        grads = []
        for param in params:
            if param.grad is not None:
                grads.append(param.grad.flatten())
        grads = torch.cat(grads)

        for i in tqdm(range(len(grads))):
            hessian_row = torch.autograd.grad(grads[i], params, retain_graph=True)
            hessian_row = torch.cat([h.flatten() for h in hessian_row])
            hessian[i] = hessian_row

        if hessian_sum_1 is None:
            hessian_sum_1 = hessian
        else:
            hessian_sum_1 += hessian
        sample_num_1 += 1

    average_hessian_1 = hessian_sum_1 / sample_num_1

    # 方式二：先累加损失，最后计算 Hessian
    total_loss = 0
    sample_num_2 = 0
    for _ in range(10):
        outputs = model(x)
        loss = criterion(outputs, y)
        total_loss += loss
        sample_num_2 += 1

    avg_loss = total_loss / sample_num_2
    optimizer.zero_grad()
    avg_loss.backward(create_graph=True)
    average_hessian_1 = average_hessian_1.to('cpu')

    params = list(model.parameters())
    param_count = sum(p.numel() for p in params)
    hessian_2 = torch.zeros(param_count, param_count).to(device)

    grads = []
    for param in params:
        if param.grad is not None:
            grads.append(param.grad.flatten())
        grads = torch.cat(grads)

        for i in tqdm(range(len(grads))):
            hessian_row = torch.autograd.grad(grads[i], params, retain_graph=True)
            hessian_row = torch.cat([h.flatten() for h in hessian_row])
            hessian_2[i] = hessian_row

    hessian_2 = hessian_2.to('cpu')
    # 比较两种方式计算的 Hessian
    equivalent = torch.allclose(average_hessian_1, hessian_2, atol=1e-5)
    if equivalent:
        print("两种计算方式得到的 Hessian 矩阵近似相等。")
    else:
        print("两种计算方式得到的 Hessian 矩阵差异较大。")

    # 可视化部分
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 绘制方式一计算的 Hessian 矩阵
    axes[0].imshow(average_hessian_1.cpu().detach().numpy(), cmap='viridis')
    axes[0].set_title('Hessian computed by Method 1')
    axes[0].axis('off')

    # 绘制方式二计算的 Hessian 矩阵
    axes[1].imshow(hessian_2.cpu().detach().numpy(), cmap='viridis')
    axes[1].set_title('Hessian computed by Method 2')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    print(average_hessian_1)
    print(hessian_2)

