"""
Classes and functions for tracking a model's optimization trajectory and computing
a low-dimensional approximation of the trajectory.
"""
from sklearn.decomposition import PCA
from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np
import pickle, os
import itertools

def get_save_root(directory='/', experiment_name=None):
    fix = experiment_name
    if experiment_name is None:
        fix = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return os.path.join(directory, experiment_name if experiment_name else datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

class TrajectoryTracker:
    """
    A TrajectoryTracker facilitates tracking the optimization trajectory of a
    DL/RL model. Trajectory trackers provide facilities for storing model parameters
    as well as for retrieving and operating on stored parameters.
    """
    imp_attr = ['trajectory']
    sign = None
    save_path = None
    trajectory = []

    def __init__(self, models=None, directory='/', experiment_name=None, reload=False, **kwargs):
        self.save_path = f'{os.path.join(directory, experiment_name if experiment_name else self.sign)}.pkl'
        if reload:
            assert os.path.exists(self.save_path)
            store = pickle.load(open(self.save_path, 'rb'))['trajectory']
            for key, value in store.items():
                setattr(self, key, value)
        else:
            assert models and len(models) > 0
            self.handle_trajectory(models, **kwargs)

    def __getitem__(self, timestep) -> np.ndarray:
        """
        Returns the position of the model from the given training timestep as a numpy array.
        :param timestep: training step of parameters to retrieve
        :return: numpy array
        """
        return self.trajectory[timestep]

    def __iter__(self) -> itertools.chain:
        """
        Returns the position of the model from the given training timestep as a numpy array.
        :return:
        """
        return itertools.chain(self.trajectory)

    def get_trajectory(self) -> list:
        """
        Returns a reference to the currently stored trajectory.
        :return: numpy array
        """
        return self.trajectory

    def save(self):
        """
        Appends the current model parameterization to the stored training trajectory.
        :param model: model object with current state of interest
        :return: N/A
        """
        pickle.dump({k: getattr(self, k) for k in self.imp_attr}, open(self.save_path, 'wb'))

    @abstractmethod
    def handle_trajectory(self, models, **kwargs):
        pass

class RandomTrajectoryTracker(TrajectoryTracker):
    """
    A ProjectingTrajectoryTracker is a tracker which applies dimensionality reduction to
    all model parameterizations upon storage. This is particularly appropriate for large
    models, where storing a history of points in the model's parameter space would be
    unfeasible in terms of memory.
    """
    imp_attr = ['trajectory', 'directions']
    sign = 'random_project_trajectory'
    directions = []
    def __init__(self, models=None, directory='/', experiment_name=None, reload=False, seed=0, directions=None):
        super().__init__(models, directory, experiment_name, reload, seed=seed, directions=directions)

    def handle_trajectory(self, models, **kwargs):
        """Project self.optim_path_matrix onto (u, v)."""
        seed = kwargs.get('seed', 0)
        directions = kwargs.get('directions')
        npvectors = []
        for model in models:
            npvectors.append(model.get_parameter_tensor().as_numpy())
        optim_path_matrix = np.vstack(npvectors)
        if directions is None:
            np.random.seed(seed)
            n_steps, n_dim = optim_path_matrix.shape
            u_gen = np.random.normal(size=n_dim)
            u = u_gen / np.linalg.norm(u_gen)
            v_gen = np.random.normal(size=n_dim)
            v = v_gen / np.linalg.norm(v_gen)  # 随机生成的两个标准化向量作为投影方向
            reduced_dirs = np.array([u, v])
            path_projection = optim_path_matrix.dot(reduced_dirs.T)
            self.directions = reduced_dirs.tolist()
        else:
            assert type(directions) == tuple
            u, v = directions
            u = u / np.linalg.norm(u)
            v = v / np.linalg.norm(v)
            reduced_dirs = np.array([u, v])
            path_projection = optim_path_matrix.dot(reduced_dirs.T)
            self.directions = reduced_dirs.tolist()
        self.trajectory = path_projection.tolist()

class PCATrajectoryTracker(TrajectoryTracker):
    imp_attr = ['trajectory', 'directions', 'pcvariances']
    directions = []
    pcvariances = 0.0
    def __init__(self, models, directory='/', experiment_name=None, reload=False, seed=0):
        super().__init__(models, directory, experiment_name, reload, ssed=seed)

    def handle_trajectory(self, models, **kwargs):
        seed = kwargs.get('seed', 0)
        np.random.seed(seed)
        npvectors = []
        for model in models:
            npvectors.append(model.get_parameter_tensor().as_numpy())
        optim_path_matrix = np.vstack(npvectors)
        pca = PCA(n_components=2, random_state=seed)
        self.trajectory = pca.fit_transform(optim_path_matrix).tolist()
        self.directions = pca.components_.tolist()
        self.pcvariances = pca.explained_variance_ratio_
