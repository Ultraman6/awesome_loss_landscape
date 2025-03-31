"""
Functions for approximating loss/return landscape in one and two dimensions.
"""
import os, sys
from tqdm import tqdm
root = os.path.dirname(os.getcwd())
sys.path.append(root)
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import torch
import copy
import pickle
from typing import Union, List
import torch.nn
import numpy as np
from landscape.model_wrapper import ModelWrapper, wrap_model
from landscape.model_parameters import rand_u_like, orthogonal_to, ModelParameters
from sklearn.decomposition import PCA
from landscape.metrics import get_metric, validate_metric, Metric
from benchmark import Trainer, train_args
from landscape.options import args, reduce_save_folder

def get_fix(**kwargs):
    return '_'.join(f'{k}={v}' for k, v in kwargs.items())

def normalize_direction(start_point, dir, normalization):
    """方向归一化处理"""
    if normalization == 'model':
        dir.model_normalize_(start_point)
    elif normalization == 'layer':
        dir.layer_normalize_(start_point)
    elif normalization == 'filter':
        dir.filter_normalize_(start_point)
    elif normalization == 'dfilter':
        for d in dir:
            d.div_(d.norm() + 1e-10)
    else:
        raise AttributeError("Unsupported normalization")

def scale_direction(start_point, dir, distance, steps, center=False):
    """方向缩放处理"""
    scale_factor = (start_point.model_norm() * distance) / steps
    dir.mul_(scale_factor / dir.model_norm())
    if center:
        dir.mul_(steps / 2)
        start_point.sub_(dir)
        dir.truediv_(steps / 2)

class Reducer:
    def __init__(self,
                 args: Namespace,
                 metric: Metric,
                 model: Union[torch.nn.Module, ModelWrapper] = None,
                 trajectory: List[Union[torch.nn.Module, ModelWrapper]] = None):
        """
        初始化任意维度损失景观算子
        参数:
        model: 基础模型实例
        trajectory: 模型参数轨迹列表
        normalization: 方向归一化方式 (filter/layer/model)
        project_method: 投影方法 (cos/lstsq)
        distance: 最大探索距离
        steps: 网格步数
        center: 是否居中
        deepcopy_model: 是否深拷贝模型
        dim: 维度
        """
        # 基本算子
        self.start_wrapper = wrap_model(model, dtype=args.dtype, ignore=args.ignore) if model else None
        self.trajectory = [wrap_model(model) for model in trajectory]
        self.device = args.device
        # 顶层参数
        self.dim = args.dim
        self.normalize = args.normalize
        self.dtype = args.dtype
        self.ignore = args.ignore
        self.signs = args.signs
        self.saves = args.saves
        self.loads = args.loads
        self.metric = metric
        self.save_root = str(args.save_root)
        os.makedirs(args.save_root, exist_ok=True)
        # 关键算子
        self.start_point = self.start_wrapper.get_module_parameters() \
            if self.start_wrapper else None
        self.directions: List[ModelParameters|None] = [None] * self.dim
        self.grid: np.array = None
        self.path: np.array = None
        # 算法参数
        self.args = args

    def iter(self, sign: str, pre_fix: str|None):
        save, load = sign in self.saves, sign in self.loads
        fix = getattr(self, f'_{sign}')(load)
        pre_fix = f"{pre_fix}-{fix}" if pre_fix else fix
        if load:
            self.load(sign, pre_fix)
        elif save:
            self.save(sign, pre_fix)

    def run(self):
        pre_fix = None
        for sign in self.signs:
            pre_fix = self.iter(sign, pre_fix)

    def _standardize_direction(self):
        for direction in self.directions:
            normalize_direction(self.start_point, direction, self.normalize)
            scale_direction(self.start_point, direction, self.args.distance,
                            self.args.steps, self.args.center)

    def _cal_point(self, point):
        """内部方法，针对单点做指标的计算"""
        model_wrapper = copy.deepcopy(self.start_wrapper)
        start_point = model_wrapper.get_module_parameters()
        for index, i in enumerate(point):
            if i % 2 == 0:
                start_point.add_(i*self.directions[index])
            else:
                start_point.sub_(i*self.directions[index])
        model_wrapper.to(self.device).eval()
        result = self.metric(model_wrapper)
        del model_wrapper
        if type(result) is tuple:
            return result[0]
        return result

    def _grid(self, load=False):
        """通用网格评估方法"""
        kwargs = {'steps': self.args.steps, 'distance': self.args.distance}
        if not load:
            data_grid = np.zeros([self.args.steps] * self.dim)
            all_points = list(product(range(self.args.steps), repeat=self.dim))
            pbar = tqdm(total=len(all_points), desc="Processing points")
            self.metric.to(self.device)
            if self.args.grid_threads > 1:
                with ThreadPoolExecutor(max_workers=self.args.grid_threads) as executor:
                    futures = {tuple(p): executor.submit(self._cal_point,p)
                               for p in all_points}
                    for point, future in futures.items():
                        data_grid[point] = future.result()
                        pbar.set_postfix({"point": point})
                        pbar.update(1)
            else:
                for point in all_points:
                    data_grid[tuple(point)] = self._cal_point(point)
                    pbar.set_postfix({"point": point})
                    pbar.update(1)
            self.grid = data_grid
        print(f"create grid with {self.dim} dims sucessfully")
        return get_fix(**kwargs)

    def _trajectory(self, load=False):
        """
        对轨迹上的每个模型进行评估
        """
        assert self.trajectory
        kwargs = {'project': self.args.project}
        if not load:
            self.path = []
            self.metric.to(self.device)
            self.start_wrapper.to(self.device)
            for model_wrapper in tqdm(self.trajectory, desc="Trajectory Creating"):
                model_wrapper.to(self.device)
                self.path.append([])
                # step1 投影
                if self.args.project == 'cos':
                    d = (model_wrapper.get_module_parameters() - self.start_point).flat_tensor()
                    for direction in self.directions:
                        dx = direction.flat_tensor()
                        self.path[-1].append(torch.dot(d, dx) / dx.norm())
                elif self.args.project == 'lstsq':
                    A = np.vstack([d.flat_tensor().numpy() for d in self.directions]).T
                    self.path[-1].extend(np.linalg.lstsq(A, d.numpy())[0].tolist())
                elif self.args.project == 'pca' and self.args.reduce != 'pca':
                    raise AttributeError(f"PCA projection requires PCA reduce not {self.args.reduce}")
                else:
                    raise NotImplementedError("Unsupported projection method")
                # step2 计算指标
                self.path[-1].append(self.metric(model_wrapper))
                diff_parameters = model_wrapper.get_module_parameters() - self.start_point
                d = diff_parameters.flat_tensor()
                model_wrapper.to('cpu')
            self.path = np.array(self.path)
            self.metric.to('cpu')
            print(f"从 {len(self.trajectory)} 个模型中创建路径成功")
        return get_fix(**kwargs)

    def _directions(self, load=False):
        """通用扰动方向方法"""
        kwargs = {'reduce': self.args.reduce, 'center': self.args.center}
        if not load:
            validate_metric(self.metric)
            if self.args.reduce == 'random':
                self.random_direction()
            elif self.args.reduce == 'inter':
                kwargs.update({'end_root': self.args.end_root})
                self.interpolate_direction(**kwargs)
            elif self.args.reduce == 'pca':
                kwargs.update({'project': self.args.project})
                self.pca_direction(**kwargs)
            elif self.args.reduce == 'eigen':
                self.eigen_direction()
            elif self.args.reduce == 'lstsq':
                self.lstsq_direction()
            else:
                raise NotImplementedError(f"Unsupported reduce: {self.args.reduce}")
            self._standardize_direction()
            print(f"创建 {self.args.reduce} 方向集成功")
        return get_fix(**kwargs)

    def interpolate_direction(self, **kwargs):
        """
        基于插值方向的多维评估
        参数:
        *end_models: 各方向终点模型
        """
        end_root = getattr(kwargs, 'end_root', None)
        end_wraps = []
        for root, dirs, files in os.walk(end_root):
            for file in files:
                file_path = str(os.path.join(root, file))
                try:
                    store = torch.load(file_path)
                    model = copy.deepcopy(self.start_wrapper)
                    model.module.load_state_dict(store['model_state_dict'])
                    end_wraps.append(model)
                except Exception as e:
                    print(f"Error loading model from {file_path}: {e}")

        assert len(end_wraps) == self.dim
        for i, end_wrap in enumerate(end_wraps):
            self.directions[i] = (end_wrap.get_module_parameters() - self.start_point) / self.args.steps

    def random_direction(self):
        """
        基于随机方向的多维评估
        """
        self.directions[0] = rand_u_like(self.start_point)
        for i in range(1, self.dim):
            self.directions[i] = orthogonal_to(self.directions[i - 1])

    def eigen_direction(self):
        """
        基于奇异值的多维评估
        """
        hessian = get_metric('eigen', self.metric.fn,
                             self.metric.inputs, self.metric.targets, top_n=self.dim)
        hessian.to(self.device)
        self.start_wrapper.to(self.device)
        _, eigvecs = hessian(self.start_wrapper)
        self.start_wrapper.to('cpu')
        self.directions = [copy.deepcopy(self.start_point) for _ in range(self.dim)]
        for dir, eigvec in zip(self.directions, eigvecs):
            dir.from_numpy(eigvec)

    def lstsq_direction(self):
        lr, bs, epoch = 0.004, 1024, 100
        self.random_direction()

        wrapper = copy.deepcopy(self.start_wrapper)
        params = wrapper.get_module_parameters()
        weight_matrix = np.array([m.get_module_parameters().flat_tensor().numpy()
                                  for m in self.trajectory])
        relative_weight_matrix = weight_matrix - params.flat_numpy()
        temp_d_list = []
        for dir in self.directions:
            temp_d_list.append(dir.flat_tensor())
        for epoch in range(epoch):
            for batch_id, (inputs, targets) in self.metric:
                matrix = [temp_d.cpu() for temp_d in temp_d_list]
                matrix = np.vstack(matrix)
                matrix = matrix.T
                grad_d_list = [torch.zeros_like(temp_d) for temp_d in temp_d_list]
                temp_grad_loss = 0
                for weight_idx in range(len(relative_weight_matrix) - 1):
                    temp_weight = relative_weight_matrix[weight_idx, :]
                    coefs = np.linalg.lstsq(matrix, temp_weight, rcond=None)[0]
                    project_weight = matrix @ coefs.T + weight_matrix[-1, :]
                    project_weight_tensor = torch.tensor(project_weight).cuda()
                    project_weight_list = project_weight_tensor / self.start_point.as_numpy()
                    params.from_numpy(project_weight_list)

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    loss = self.metric(wrapper, inputs=inputs, targets=targets)
                    loss.backward()
                    origin_loss = self.metric(self.trajectory[weight_idx],
                                              inputs=inputs, targets=targets)
                    temp_grad_loss += (loss.item() - origin_loss[weight_idx]) ** 2

                    grad_vector = wrapper.get_grads()
                    grad_d_list = [grad_d + 2 * (loss.item() - origin_loss[weight_idx]) * coefs[i] * grad_vector
                                   for i, grad_d in enumerate(grad_d_list)]

                print('epoch: {}   batch: {}   loss: {}'.format(epoch, batch_id, temp_grad_loss))
                for i, grad_d in enumerate(grad_d_list):
                    temp_d_list[i] -= grad_d * lr

        self.directions = [dir.from_tensor(d) for dir, d in zip(self.directions, temp_d_list)]

    def pca_direction(self, **kwargs):
        """
        基于PCA轨迹的多维评估
        参数:
        trajectory: 模型参数轨迹列表
        project: 是否将初始点设置为 pca mean
        """
        assert not self.trajectory
        project = getattr(kwargs, 'project', None)
        optim_path_matrix = np.vstack([model.get_parameter_tensor().flat_numpy() for model in self.trajectory])
        pca = PCA(n_components=self.dim)
        pca.fit(optim_path_matrix)
        start_wrapper = wrap_model(self.start_wrapper)
        if project not in ['cos', 'lstsq']:  # pca 投影与起点
            start_wrapper.copy_from_numpy(np.array(pca.mean_))
            self.path = pca.transform(optim_path_matrix).tolist()
            self.start_point = start_wrapper.get_module_parameters()
        for i in range(self.dim):
            self.directions[i] = start_wrapper.copy_from_numpy(pca.components_[i])

    def save(self, attr_name: str, fix: str):
        """保存算子状态到文件"""
        file_root = os.path.join(self.save_root, attr_name)
        os.makedirs(file_root, exist_ok=True)
        file_path = os.path.join(file_root, f'{fix}.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(getattr(self, attr_name), f)
        print(f"保存 {attr_name} 从 {file_path} 成功。")

    def load(self, attr_name: str, fix: str):
        """从文件加载算子状态"""
        file_path = os.path.join(self.save_root, attr_name, f'{fix}.pkl')
        try:
            with open(file_path, 'rb') as f:
                setattr(self, attr_name, pickle.load(f))
            print(f"装载 {attr_name} 从 {file_path} 成功")
        except FileNotFoundError:
            raise ValueError(f"文件 {file_path} 未找到。")

def load_reducer(train_args: Namespace, args: Namespace) -> Reducer:
    model, trajectory, criterion, data = None, None, None, None
    trainer = Trainer(train_args)
    if args.model_file:
        trajectory = []
        model = pickle.load(open(args.model_file, 'rb'))
    if args.model_folder:
        for root, dirs, files in os.walk(args.model_folder):
            for file in files:
                trajectory.append(torch.load(str(os.path.join(root, file))))

    if args.data_file:
        data = torch.load(args.data_file)
    if args.criterion_file:
        criterion = torch.load(args.criterion_file)

    kwargs = {
        'begin_epoch': args.begin_epoch,
        'end_epoch': args.end_epoch,
        'center_epoch': args.center_epoch
    }

    if not model:
        model = trainer.get_checkpoints(**kwargs)['model']
    if not trajectory:
        trajectory = trainer.get_checkpoints(**kwargs)['trajectory']
    if not criterion:
        criterion = trainer.criterion
    if not data:
        data = trainer.data.eval_data(cls_idxes=args.cls_idxes) \
            if args.eval else trainer.data.train_data(cls_idxes=args.cls_idxes)
    inputs, targets = data
    metric = get_metric(args.metric, criterion, inputs, targets, batch_size=args.metric_bs)
    return Reducer(args, metric, model=model, trajectory=trajectory)

if __name__ == '__main__':
    args.save_root = os.path.join(train_args.save_root, reduce_save_folder(args))
    reducer = load_reducer(train_args, args)
    reducer.run()
    # for i in range(10):
    #     args.cls_idxes = (i,)
    #     args.save_root = os.path.join(train_args.save_root, reduce_save_folder(args))
    #     reducer = load_reducer(train_args, args)
    #     reducer.run()