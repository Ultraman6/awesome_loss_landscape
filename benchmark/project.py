import numpy as np
import os
import pickle
import cne
import umap

def neg_t_sne_visualization(data, dataset_name="mnist", k=15, seed=0, n_epochs=500, n_noise=5, batch_size=1024, rescale=1.0, anneal_lr=True, momentum=0.0, lr_min_factor=0.0, clamp_low=1e-10, init_type="EE", optimizer="sgd", loss_mode="neg_sample"):
    """
    Neg - t - SNE 可视化接口
    :param data: 输入的数据
    :param dataset_name: 数据集名称
    :param k: 近邻数量
    :param seed: 随机种子
    :param n_epochs: 训练轮数
    :param n_noise: 负样本数量
    :param batch_size: 批次大小
    :param rescale: 初始化缩放因子
    :param anneal_lr: 是否退火学习率
    :param momentum: 动量
    :param lr_min_factor: 最小学习率因子
    :param clamp_low: 下限值
    :param init_type: 初始化类型
    :param optimizer: 优化器
    :param loss_mode: 损失模式
    :return: 降维后的数据
    """
    root_path = get_path("data")
    _, _, sknn_graph, pca2 = load_dataset(root_path, dataset_name, k=k)

    # 初始化
    init = pca2
    if rescale:
        init = pca2 / np.std(pca2[:, 0]) * rescale

    if init_type == "random":
        np.random.seed(seed)
        init = np.random.randn(len(data), 2)
        if rescale:
            init = init / np.std(init) * rescale
        init_str = f"random_rescale_{rescale}"
    elif init_type == "pca":
        init_str = f"pca_rescale_{rescale}"
    elif init_type == "EE":
        file_name_init = os.path.join(root_path,
                                      dataset_name,
                                      f"cne_{loss_mode}_n_noise_{5}_n_epochs_{250}_init_pca_rescale_{rescale}_bs_{batch_size}"
                                      f"_optim_{optimizer}_anneal_lr_{anneal_lr}_lr_min_factor_{lr_min_factor}_momentum_{momentum}_param_{False}_clamp_low_{clamp_low}_seed_{seed}.pkl"
                                      )
        try:
            with open(file_name_init, "rb") as file:
                embedder_init_cne = pickle.load(file)
        except FileNotFoundError:
            logger = cne.callbacks.Logger(log_embds=True,
                                          log_norms=True,
                                          log_kl=True,
                                          graph=sknn_graph,
                                          n=len(data) if False else None)
            embedder_init = cne.CNE(loss_mode=loss_mode,
                                    parametric=False,
                                    negative_samples=5,
                                    n_epochs=250,
                                    batch_size=batch_size,
                                    on_gpu=True,
                                    print_freq_epoch=100,
                                    print_freq_in_epoch=None,
                                    callback=logger,
                                    optimizer=optimizer,
                                    momentum=momentum,
                                    save_freq=1,
                                    anneal_lr=anneal_lr,
                                    lr_min_factor=lr_min_factor,
                                    clamp_low=clamp_low,
                                    seed=seed,
                                    loss_aggregation="sum",
                                    force_resample=True
                                    )
            embedder_init.fit(data, init=init, graph=sknn_graph)
            embedder_init_cne = embedder_init.cne
            with open(file_name_init, "wb") as file:
                pickle.dump(embedder_init_cne, file, pickle.HIGHEST_PROTOCOL)
        init = embedder_init_cne.callback.embds[-1]
        init_str = "EE"

    nbs_noise_in_estimator = get_noise_in_estimator(len(data), 5, dataset_name)
    noise_in_estimator = nbs_noise_in_estimator[0]

    file_name = os.path.join(root_path,
                             dataset_name,
                             f"cne_{loss_mode}_n_noise_{n_noise}_noise_in_estimator_{noise_in_estimator}_n_epochs_{n_epochs}_init_{init_str}_bs_{batch_size}"
                             f"_optim_{optimizer}_anneal_lr_{anneal_lr}_lr_min_factor_{lr_min_factor}_momentum_{momentum}_param_{False}_clamp_low_{clamp_low}_seed_{seed}.pkl"
                             )
    try:
        with open(file_name, "rb") as file:
            embedder_cne = pickle.load(file)
    except FileNotFoundError:
        logger = cne.callbacks.Logger(log_embds=True,
                                      log_norms=True,
                                      log_kl=True,
                                      graph=sknn_graph,
                                      n=len(data) if False else None)
        embedder = cne.CNE(loss_mode=loss_mode,
                           parametric=False,
                           negative_samples=n_noise,
                           n_epochs=n_epochs,
                           batch_size=batch_size,
                           on_gpu=True,
                           print_freq_epoch=100,
                           print_freq_in_epoch=None,
                           callback=logger,
                           optimizer=optimizer,
                           momentum=momentum,
                           save_freq=1,
                           anneal_lr=anneal_lr,
                           noise_in_estimator=noise_in_estimator,
                           lr_min_factor=lr_min_factor,
                           clamp_low=clamp_low,
                           seed=seed,
                           loss_aggregation="sum",
                           force_resample=True
                           )
        embedder.fit(data, init=init, graph=sknn_graph)
        embedder_cne = embedder.cne
        with open(file_name, "wb") as file:
            pickle.dump(embedder_cne, file, pickle.HIGHEST_PROTOCOL)

    return embedder_cne.callback.embds[-1]


def umap_visualization(data, dataset_name="mnist", k=15, seed=0, n_epochs=750, anneal_lr=True, rescale=1.0, n_noise=5, eps=1.0, a=1.0, b=1.0):
    """
    UMAP 可视化接口
    :param data: 输入的数据
    :param dataset_name: 数据集名称
    :param k: 近邻数量
    :param seed: 随机种子
    :param n_epochs: 训练轮数
    :param anneal_lr: 是否退火学习率
    :param rescale: 初始化缩放因子
    :param n_noise: 负样本数量
    :param eps: 优化技巧的参数
    :param a: UMAP 参数
    :param b: UMAP 参数
    :return: 降维后的数据
    """
    root_path = get_path("data")
    _, _, sknn_graph, pca2 = load_dataset(root_path, dataset_name, k)

    pca_rescaled = pca2 / np.std(pca2[:, 0]) * rescale if rescale else pca2

    filename = os.path.join(root_path, dataset_name,
                            f"umap_bin_k_{k}_n_epochs_{n_epochs}_anneal_lr_{anneal_lr}_eps_{eps}_seed_{seed}_a_{a}_b_{b}_init_pca_rescaled_{rescale}.pkl")
    try:
        with open(filename, "rb") as file:
            umapper = pickle.load(file)
    except FileNotFoundError:
        umapper = umap.UMAP(n_neighbors=k,
                            a=a,
                            b=b,
                            n_epochs=n_epochs,
                            negative_sample_rate=n_noise,
                            log_embeddings=True,
                            random_state=seed,
                            init=pca_rescaled,
                            graph=sknn_graph,
                            verbose=True,
                            anneal_lr=anneal_lr,
                            eps=eps)
        umapper.fit_transform(data)
        with open(filename, "wb") as file:
            pickle.dump(umapper, file, pickle.HIGHEST_PROTOCOL)

    return umapper.embedding_
