import os, sys
root = os.path.dirname(os.getcwd())
sys.path.append(root)
from argparse import Namespace
import pickle
import numpy as np
from matplotlib import pyplot as plt
from visualization.paraview import compute_persistence_barcode, compute_merge_tree, compute_merge_tree_planar
from benchmark import train_args
from landscape import Reducer, reduce_args, load_reducer
from options import args

def get_fix(**kwargs):
    return '_'.join(f'{k}={v}' for k, v in kwargs.items())

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

class Visualizer:
    def __init__(self, reducer: Reducer, args: Namespace):
        if not args.coords_file and not args.values_file:
            self.coords = pickle.load(open(args.coords_file, 'rb'))
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
            result = getattr(self, self.args.visual_mode)(**kwargs)
            print(result)
        else:
            self.args.output_path = os.path.join(self.args.output_path, f"{self.args.visual_mode}.pdf")
            getattr(self, self.args.visual_mode)()
            plt.savefig(self.args.output_path)

    def line(self):
        """
        一维线图可视化
        """
        if self.reducer.dim != 1:
            raise ValueError("一维线图可视化仅适用于一维数据")
        steps = self.reducer.args.steps
        x = np.linspace(0, self.reducer.args.distance, steps)
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
        steps = self.reducer.args.steps
        x = np.linspace(0, self.reducer.args.distance, steps)
        y = np.linspace(0, self.reducer.args.distance, steps)
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
        steps = self.reducer.args.steps
        x = np.linspace(0, self.reducer.args.distance, steps)
        y = np.linspace(0, self.reducer.args.distance, steps)
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

if __name__ == "__main__":
    reducer = load_reducer(train_args, reduce_args)
    visualizer = Visualizer(reducer, args)
    visualizer.run()
