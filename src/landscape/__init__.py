"""Dimensionality reduction and loss grid computation."""
import os.path
import pickle
from abc import abstractmethod
import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from src.util import min_max_hessian_eigs

def _load_models(model, args):
    param_list = []
    ckpt_root = os.path.join(args.save_root, 'checkpoints')
    for epoch in range(args.start_epoch, args.max_epoch + args.save_epoch, args.save_epoch):
        ckpt_path = os.path.join(ckpt_root, f"epoch={epoch}.ckpt")
        store = torch.load(ckpt_path)
        model.load_state_dict(store['state_dict'])
        param_list.append(model.get_flat_params())
    return param_list

class DimReduction:
    def __init__(self, model, args):
        self.optim_path_matrix = self._transform(_load_models(model, args))
        self.n_steps, self.n_dim = self.optim_path_matrix.shape
        self.reduce=args.reduce
        self.dir_path = f"{os.path.join(args.plot_root, f'{args.reduce}_direction')}.pkl"
        self.seed=args.seed
        self.custom_directions = None

    def reduce(self, save=True):
        """Perform the reduction on the target matrix.

        Raises:
            Exception: reduction method not recognized.

        Returns:
            The reduced matrix.
        """
        if os.path.exists(self.dir_path):
            return pickle.load(open(self.dir_path, "rb"))

        if self.reduce == "pca":
            store = self.pca()
        elif self.reduce == "random":
            store = self.reduce_to_random_directions()
        else:
            raise Exception(f"Unrecognized reduction method {self.reduce}")

        if save:
            pickle.dump(store, open(self.dir_path, "wb"))

        return store

    def pca(self):
        """Perform PCA on the input matrix.

        Returns:
            A dict of values including the full-dimensional path, the 2D path,
            the reduced directions, and percentage of variance explained by the
            directions.
        """
        pca = PCA(n_components=2, random_state=self.seed)
        path_2d = pca.fit_transform(self.optim_path_matrix)
        reduced_dirs = pca.components_
        assert path_2d.shape == (self.n_steps, 2)
        return {
            "optim_path": self.optim_path_matrix,
            "path_2d": path_2d,
            "reduced_dirs": reduced_dirs,
            "pcvariances": pca.explained_variance_ratio_,
        }

    def reduce_to_random_directions(self):
        """Produce 2 random flat unit vectors of dim <dim_params> as the directions.

        Since 2 high-dimensional vectors are almost always orthogonal,
        it's no problem to use them as the axes for the 2D slice of
        loss landscape.
        """
        print("Generating random axes...")
        # Generate 2 random unit vectors (u, v)
        if self.seed:
            print(f"seed={self.seed}")
            np.random.seed(self.seed)
        u_gen = np.random.normal(size=self.n_dim)
        u = u_gen / np.linalg.norm(u_gen)
        v_gen = np.random.normal(size=self.n_dim)
        v = v_gen / np.linalg.norm(v_gen)  # 随机生成的两个标准化向量作为投影方向
        return self._project(np.array([u, v]))

    def reduce_to_custom_directions(self):
        """Manually pick two direction vectors dir0, dir1 of dim <dim_params>.

        Use them as the axes for the 2D slice of loss landscape.
        """
        print("Using custom axes...")  # 利用三个模型参数自定义两个投影方向
        dir0, dir1 = self.custom_directions
        dir0_exists = dir0 is not None
        dir1_exists = dir1 is not None
        if not (dir0_exists and dir1_exists):
            raise Exception(
                "Custom directions not provided, please provide 2 vectors of "
                f"dim={self.n_dim}"
            )
        # Normalize given direction vectors
        u = dir0 / np.linalg.norm(dir0)
        v = dir1 / np.linalg.norm(dir1)
        # Transform all step params into the coordinates of (u, v)
        return self._project(np.array([u, v]))

    def _project(self, reduced_dirs):
        """Project self.optim_path_matrix onto (u, v)."""
        path_projection = self.optim_path_matrix.dot(reduced_dirs.T)
        assert path_projection.shape == (self.n_steps, 2) # 投影就是内积操作
        return {
            "optim_path": self.optim_path_matrix,
            "path_2d": path_projection,
            "reduced_dirs": reduced_dirs,
        }

    def _transform(self, model_params):
        npvectors = []
        for tensor in model_params:
            npvectors.append(np.array(tensor.cpu()))
        return np.vstack(npvectors)

class Grid:
    signal=None
    def __init__(
        self,
        model,
        data,
        optim_path,
        optim_path_2d,
        directions,
        args,
    ):
        self.dir0, self.dir1 = directions  # 就是存储两个投影方向向量的元组
        self.optim_path = optim_path
        self.optim_path_2d = optim_path_2d  # 存储投影优化路径
        self.optim_point = optim_path[-1]  # 存储原始最优点（最终点）
        self.optim_point_2d = optim_path_2d[-1]  # 存储投影最优点
        self.res = args.res
        self.margin = args.margin
        self.alpha = self._compute_stepsize()  # 放缩投影步长
        self.params_grid = self.build_params_grid()  # 原始网格坐标系
        self.grid_path = f"{os.path.join(args.plot_root, f'{args.reduce}_{self.signal}grid_[{args.res, args.margin}]')}.pkl"

        if os.path.exists(self.grid_path):
            self.loss_values_2d, self.argmin, self.loss_min = pickle.load(open(self.grid_path, "rb"))
            print(f"{self.signal} grid loaded from {self.params_grid}")
        else:  # 获取投影损失网络
            self.loss_values_2d, self.argmin, self.loss_min = self.compute_values(model, data, args.tqdm)

            if args.save_grid:
                loss_2d_tup = (self.loss_values_2d, self.argmin, self.loss_min)
                pickle.dump(loss_2d_tup, open(self.grid_path, "wb"))
                print(f"{self.signal} grid saved at {self.grid_path}.")

        self.coords = self._convert_coords(self.res, self.alpha)  # 投影网格坐标系
        self.true_optim_point = self.indices_to_coords(self.argmin, self.res, self.alpha)

    # 生成网格
    def build_params_grid(self):
        grid = []  # 以原始最优点为中心，生成2x2网格
        for i in range(-self.res, self.res):
            row = []
            for j in range(-self.res, self.res):
                w_new = (
                        self.optim_point.cpu()
                        + i * self.alpha * self.dir0
                        + j * self.alpha * self.dir1
                )
                row.append(w_new)
            grid.append(row)
        assert (grid[self.res][self.res] == self.optim_point.cpu()).all()
        return grid

    # 计算网格损失
    def compute_values(self, model, data, tqdm_disable=False):
        values = []
        n = len(self.params_grid)
        m = len(self.params_grid[0])
        _min = float("inf")
        argmin = ()
        print("Generating loss values for the contour landscape...")
        with tqdm(total=n * m, disable=tqdm_disable) as pbar:
            for i in range(n):
                loss_row = []
                for j in range(m):
                    w_ij = torch.Tensor(self.params_grid[i][j].float())
                    model.init_from_flat_params(w_ij)
                    _val = self._compute(model, data)
                    if _val < _min:
                        loss_min = _val
                        argmin = (i, j)
                    loss_row.append(_val)
                    pbar.update(1)
                values.append(loss_row)
        print(f"{self.signal} values generated.")
        return np.array(values).T, argmin, loss_min

    @abstractmethod
    def _compute(self, model, data):
        pass

    def _convert_coord(self, i, ref_point_coord, alpha):
        return i * alpha + ref_point_coord

    def _convert_coords(self, res, alpha):
        converted_coord_xs = []
        converted_coord_ys = []
        for i in range(-res, res):
            x = self._convert_coord(i, self.optim_point_2d[0], alpha)
            y = self._convert_coord(i, self.optim_point_2d[1], alpha)
            converted_coord_xs.append(x)
            converted_coord_ys.append(y)
        return np.array(converted_coord_xs), np.array(converted_coord_ys)

    def indices_to_coords(self, indices, res, alpha):
        grid_i, grid_j = indices
        i, j = grid_i - res, grid_j - res
        x = i * alpha + self.optim_point_2d[0]
        y = j * alpha + self.optim_point_2d[1]
        return x, y

    def _compute_stepsize(self):
        dist_2d = self.path_2d[-1] - self.path_2d[0]  # 计算开始<->结束的投影差距
        dist = (dist_2d[0] ** 2 + dist_2d[1] ** 2) ** 0.5  # 计算投影距离
        return dist * (1 + self.margin) / self.res  # 获得每步距离

class LossGrid(Grid):
    signal='loss'
    def __int__(
        self,
        optim_path,
        model,
        data,
        path_2d,
        directions,
        args,
        log=True
    ):
        super().__init__(optim_path, model, data, path_2d, directions, args)
        if log:
            self.loss_values_2d = np.log(self.loss_values_2d)

    def _compute(self, model, data):
        X, y = data
        y_pred = model(X)
        return model.loss_fn(y_pred, y).item()

class EigenGrid(Grid):
    signal='eigen'
    def __int__(
        self,
        optim_path,
        model,
        data,
        path_2d,
        directions,
        args,
        abs=True
    ):
        super().__init__(optim_path, model, data, path_2d, directions, args)
        self.abs = abs
        if abs:
            self.grid_path = f"{os.path.join(args.save, args.plot_path, f'{args.reduce}_grid')}.pkl"

    def _compute(self, model, data):
        maxeig, mineig, iter_count = min_max_hessian_eigs(model, data)
        if self.abs:
            return np.abs(np.divide(mineig, maxeig))
        return np.divide(mineig, maxeig)