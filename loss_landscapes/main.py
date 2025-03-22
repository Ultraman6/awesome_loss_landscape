"""
Functions for approximating loss/return landscapes in one and two dimensions.
"""
import os
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import torch
import copy
import pickle
from typing import Union, List, Callable
import torch.nn
import numpy as np
from matplotlib import pyplot as plt
from loss_landscapes import args
import benchmarks
from loss_landscapes.wrapper.model_wrapper import ModelWrapper, wrap_model
from loss_landscapes.wrapper.model_parameters import rand_u_like, orthogonal_to, ModelParameters
from sklearn.decomposition import PCA
from loss_landscapes.metrics import HessianEvaluator
from paraview import compute_persistence_barcode, compute_merge_tree, compute_merge_tree_planar
from benchmarks.main import Runner
from loss_landscapes.metrics.sl_metrics import Loss

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

def load_coords_values(grid: np.ndarray):
    coords, values = [], []
    # 获取网格的维度
    dimensions = grid.shape
    # 遍历网格的每个元素
    for index in np.ndindex(dimensions):
        # 记录当前元素的坐标
        coords.append(list(index))
        # 记录当前元素的值
        values.append(grid[index])
    return np.array(coords), np.array(values)

class Reducer:
    def __init__(self,
                 args: Namespace,
                 metric: Callable,
                 model: Union[torch.nn.Module, ModelWrapper] = None,
                 trajectory: List[Union[torch.nn.Module, ModelWrapper]] = ()):
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
        self.deepcopy = args.deepcopy
        self.model = copy.deepcopy(model) if self.deepcopy else model
        self.trajectory = [copy.deepcopy(model) if self.deepcopy else model for model in trajectory]
        # 顶层参数
        self.dim = args.dim
        self.normalize = args.normalize
        self.dtype = args.dtype
        self.ignore = args.ignore
        self.save_root = args.save_root
        self.signs = args.signs
        self.saves = args.saves
        self.loads = args.loads
        # 方向参数
        self.center = args.center
        self.pca_path = args.pca_path
        self.reduce = args.reduce
        self.end_root = args.end_root
        self.eigen_mode = 'real'
        self.eigen_maxIter = 100
        self.eigen_tol = 1e-3
        # 网格参数
        self.grid_threads = args.grid_threads
        self.steps = args.steps
        self.distance = args.distance
        # 路径参数
        self.project = args.project
        # 关键算子
        self.metric = metric
        self.start_point = self._get_model_wrapper(self.model).get_module_parameters() if self.model else None
        self.directions: List[ModelParameters|None] = [None] * self.dim
        self.grid: np.array = None
        self.path: np.array = None

    def iter(self, sign: str, pre_fix: str|None):
        save, load = sign in self.saves, sign in self.loads
        fix = getattr(self, sign)(load)
        pre_fix = f"{pre_fix}-{fix}" if not pre_fix else fix
        if load:
            self.load(sign, pre_fix)
        elif save:
            self.save(sign, pre_fix)

    def run(self):
        pre_fix = None
        for sign in self.signs:
            pre_fix = self.iter(sign, pre_fix)

    def _validate_metric(self):
        """验证metric函数有效性"""
        if not callable(self.metric):
            raise ValueError("metric must be a callable function")

    def _get_model_wrapper(self, model: Union[torch.nn.Module, ModelWrapper]):
        """统一模型包装处理"""
        return wrap_model(copy.deepcopy(model), dtype=self.dtype, ignore=self.ignore) if self.deepcopy else model

    def _standardize_direction(self):
        for direction in self.directions:
            normalize_direction(self.start_point, direction, self.normalize)
            scale_direction(self.start_point, direction, self.distance, self.steps, self.center)

    def _cal_point(self, point, model_wrapper):
        """内部方法，针对单点做指标的计算"""
        for index, i in enumerate(point):
            if i % 2 == 0:
                self.start_point.add_(self.directions[index])
            else:
                self.start_point.sub_(self.directions[index])

        return self.metric(model_wrapper)

    def _grid(self, load=False):
        """通用网格评估方法"""
        kwargs = {'steps': self.steps, 'distance': self.distance}
        if not load:
            data_grid = np.zeros([self.steps] * self.dim)
            model_wrapper = self._get_model_wrapper(self.model)
            all_points = product(range(self.steps), repeat=self.dim)
            if self.grid_threads > 1:
                with ThreadPoolExecutor(max_workers=self.grid_threads) as executor:
                    futures = {tuple(p): executor.submit(self._cal_point, p, copy.deepcopy(model_wrapper))
                               for p in all_points}
                    for point, future in futures.items():
                        data_grid[point] = future.result()
            else:
                start_point_parameters = copy.deepcopy(self.start_point._get_parameters())
                for point in all_points:
                    data_grid[tuple(point)] = self._cal_point(point, model_wrapper)
                self.start_point._set_parameters(start_point_parameters)
            self.grid = data_grid

        return get_fix(**kwargs)

    def _trajectory(self, load=False):
        """
        对轨迹上的每个模型进行评估
        """
        kwargs = {'project': self.steps}
        if not load:
            self.path = []
            for model in self.trajectory:
                self.path.append([])
                # step1 投影
                if self.project == 'cos':
                    for direction in self.directions:
                        dx = direction.flat_tensor()
                        self.path[-1].append(torch.dot(d, dx) / dx.norm())
                elif self.project == 'lstsq':
                    A = np.vstack([dir.flat_tensor().numpy() for dir in self.directions]).T
                    self.path[-1].extend(np.linalg.lstsq(A, d.numpy())[0].tolist())
                else:
                    raise NotImplementedError("Unsupported projection method")
                # step2 计算指标
                model_wrapper = self._get_model_wrapper(model)
                self.path[-1].append(self.metric(model_wrapper))
                diff_parameters = model_wrapper.get_module_parameters() - self.start_point
                d = diff_parameters.flat_tensor()
                self.path = np.array(self.path)

        return get_fix(**kwargs)

    def _directions(self, load=False):
        """通用扰动方向方法"""
        kwargs = {'reduce': self.reduce, 'center': self.center}
        if not load:
            if self.reduce == 'random':
                self.random_direction()
            elif self.reduce == 'inter':
                kwargs.update({'end_root': self.end_root})
                self.interpolate_direction(**kwargs)
            elif self.reduce == 'pca':
                kwargs.update({'project': self.project})
                self.pca_direction(**kwargs)
            elif self.reduce == 'eigen':
                kwargs.update({'mode': self.eigen_mode, 'maxIter': self.eigen_maxIter, 'tol': self.eigen_tol})
                self.eigen_direction(**kwargs)
            else:
                raise NotImplementedError(f"Unsupported reduce: {self.reduce}")
        return get_fix(**kwargs)

    def interpolate_direction(self, **kwargs):
        """
        基于插值方向的多维评估
        参数:
        *end_models: 各方向终点模型
        """
        end_root = getattr(kwargs, 'end_root', None)
        end_models = []
        for root, dirs, files in os.walk(end_root):
            for file in files:
                file_path = str(os.path.join(root, file))
                try:
                    store = torch.load(file_path)
                    model = copy.deepcopy(self.model)
                    model.load_state_dict(store['model_state_dict'])
                    end_models.append(model)
                except Exception as e:
                    print(f"Error loading model from {file_path}: {e}")

        assert len(end_models) == self.dim
        for i, end_model in enumerate(end_models):
            end_wrapper = self._get_model_wrapper(end_model)
            self.directions[i] = (end_wrapper.get_module_parameters() - self.start_point) / self.steps
        self._standardize_direction()

    def random_direction(self):
        """
        基于随机方向的多维评估
        """
        self._validate_metric()
        self.directions[0] = rand_u_like(self.start_point)
        for i in range(1, self.dim):
            self.directions[i] = orthogonal_to(self.directions[i - 1])
        self._standardize_direction()

    def eigen_direction(self, **kwargs):
        """
        基于奇异值的多维评估
        """
        maxIter = getattr(kwargs, 'maxIter', 100)
        tol = getattr(kwargs, 'tol', 1e-3)
        self._validate_metric()
        _, eigen_vectors = HessianEvaluator(self.dim, maxIter, tol, **self.metric.__dict__)
        self.directions = eigen_vectors
        self._standardize_direction()

    def pca_direction(self, **kwargs):
        """
        基于PCA轨迹的多维评估
        参数:
        trajectory: 模型参数轨迹列表
        project: 是否将初始点设置为 pca mean
        """
        project = getattr(kwargs, 'project', None)
        # 提取轨迹参数
        optim_path_matrix = np.vstack([model.get_parameter_tensor().as_numpy() for model in self.trajectory])
        # PCA降维
        pca = PCA(n_components=self.dim)
        pca.fit(optim_path_matrix)
        # 构建方向向量
        start_wrapper = self._get_model_wrapper(self.model)
        if project not in ['cos', 'lstsq']: # pca 投影与起点
            start_wrapper.copy_from_numpy(np.array(pca.mean_))
            self.path = pca.transform(optim_path_matrix).tolist()
            self.start_point = start_wrapper.get_module_parameters()
        for i in range(self.dim):
            self.directions[i] = start_wrapper.copy_from_numpy(pca.components_[i])
        self._standardize_direction()

    def save(self, attr_name: str, fix: str):
        """保存算子状态到文件"""
        file_path = os.path.join(self.save_root, attr_name, fix)
        with open(file_path, 'wb') as f:
            pickle.dump(getattr(self, attr_name), f)

    def load(self, attr_name: str, fix: str):
        """从文件加载算子状态"""
        file_path = os.path.join(self.save_root, attr_name, fix)
        try:
            with open(file_path, 'rb') as f:
                setattr(self, attr_name, pickle.load(f))
        except FileNotFoundError:
            print(f"文件 {file_path} 未找到。")

class Visualizer:
    def __init__(self, reducer: Reducer, args: Namespace):
        if not args.coords_file:
            self.coords = pickle.load(open(args.coords_file, 'rb'))
        elif not args.values_file:
            self.values = pickle.load(open(args.values_file, 'rb'))
        elif not args.grid_file:
            self.coords, self.values = load_coords_values(pickle.load(open(args.grid_file, 'rb')))
        else:
            self.coords, self.values = load_coords_values(reducer.grid)

        self.reducer = reducer
        self.dim = self.coords[0].shape[0]
        self.args = args

    def run(self):
        if self.args.visual_mode in ['persistence_barcode', 'merge_tree', 'merge_tree_planar']:
            kwargs = {
                "dim": self.dim,
                "vtk_format": self.args.vtk_format,
                "graph_kwargs": self.args.graph_kwargs,
                "n_neighbors": self.args.n_neighbors,
                "persistence_threshold": self.args.persistence_threshold,
                "threshold_is_absolute": self.args.threshold_is_absolute
            }
            fix = get_fix(**kwargs)
            self.args.output_path = os.path.join(self.args.output_path, f"{self.args.visual_mode}-{fix}")
            kwargs.update({"loss_coords": self.coords, "loss_values": self.values, "output_path": self.args.output_path})
            result = getattr(self.reducer, self.args.visual_mode)(**kwargs)
            print(result)
        else:
            # kwargs = {}
            # fix = get_fix(**kwargs)
            self.args.output_path = os.path.join(self.args.output_path, f"{self.args.visual_mode}")
            getattr(self.reducer, self.args.visual_mode)()
            plt.savefig(self.args.output_path)

    def line(self):
        """
        一维线图可视化
        """
        if self.reducer.dim != 1:
            raise ValueError("一维线图可视化仅适用于一维数据")
        steps = self.reducer.steps
        x = np.linspace(0, self.reducer.distance, steps)
        y = self.reducer.grid.flatten()

        plt.figure()
        plt.plot(x, y)
        plt.xlabel('Distance')
        plt.ylabel('Loss')
        plt.title('1D Loss Landscape')
        plt.show()

    def contour(self):
        """
        二维等高线图可视化
        """
        if self.reducer.dim != 2:
            raise ValueError("二维等高线图可视化仅适用于二维数据")
        steps = self.reducer.steps
        x = np.linspace(0, self.reducer.distance, steps)
        y = np.linspace(0, self.reducer.distance, steps)
        X, Y = np.meshgrid(x, y)
        Z = self.reducer.grid

        plt.figure()
        plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.colorbar()
        plt.xlabel('Direction 1')
        plt.ylabel('Direction 2')
        plt.title('2D Loss Landscape (Contour Plot)')
        plt.show()

    def heatmap(self):
        """
        二维热力图可视化
        """
        if self.reducer.dim != 2:
            raise ValueError("二维热力图可视化仅适用于二维数据")
        Z = self.reducer.grid

        plt.figure()
        plt.imshow(Z, cmap='viridis', origin='lower')
        plt.colorbar()
        plt.xlabel('Direction 1')
        plt.ylabel('Direction 2')
        plt.title('2D Loss Landscape (Heatmap)')
        plt.show()

    def surface(self):
        """
        二维数据的 3D 图可视化
        """
        if self.reducer.dim != 2:
            raise ValueError("二维数据的 3D 图可视化仅适用于二维数据")
        steps = self.reducer.steps
        x = np.linspace(0, self.reducer.distance, steps)
        y = np.linspace(0, self.reducer.distance, steps)
        X, Y = np.meshgrid(x, y)
        Z = self.reducer.grid

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_xlabel('Direction 1')
        ax.set_ylabel('Direction 2')
        ax.set_zlabel('Loss')
        ax.set_title('2D Loss Landscape (3D Plot)')
        plt.show()

    def persistence_barcode(self, **kwargs):
        result = compute_persistence_barcode(**kwargs)
        return result

    def merge_tree(self,  **kwargs):
        result = compute_merge_tree(**kwargs)
        return result

    def merge_tree_planar(self,  **kwargs):
        result = compute_merge_tree_planar(**kwargs)
        return result

if __name__ == '__main__':
    trainer = Runner(benchmarks.args)
    result = trainer.load_checkpoint()
    if args.eval:
        trainer.data.eval = True
        inputs, targets = trainer.data.eval_data()
    else:
        inputs, targets = trainer.data.eval_data()
    metric = Loss(trainer.criterion, inputs, targets)
    result = trainer.load_checkpoint()
    reducer =  Reducer(args, metric, model=result['model'], trajectory=result['trajectory'])