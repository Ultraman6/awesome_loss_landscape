"""The main module with the src API.
Conceptual steps to produce the animation:
1. Load datasets
2. Create a pytorch lightning models
3. Record the parameters during training
4. Use PCA's top 2 PCs (or any 2 directions) to project the parameters to 2D
5. Collect the values in 2D:
    a. A list of 2D values as the trajectory obtained by projecting the
       parameters down to the 2D space.
    b. A 2D slice of the loss landscape (loss grid) that capture (a) with some
       adjustments for visual aesthetics.
"""
import os.path
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from src.models import load_model
from src.datasets import load_data


def train(args):
    torch.manual_seed(args.seed)
    # load data
    data = load_data(args=args)
    train_loader = data.train_dataloader()
    # load model
    ckpt_path = None
    if args.resume_epoch > -1:
        ckpt_path = os.path.join(args.save_root, 'checkpoints', f"epoch={args.resume_epoch}.ckpt")
    model = load_model(args=args)
    # load callback
    epoch_callback, best_callback = None, None
    if args.save_epoch > 0:
        epoch_callback = ModelCheckpoint(
            monitor='epoch',
            dirpath=os.path.join(args.save_root, 'checkpoints'),
            filename='{epoch}',
            save_top_k=-1,
            every_n_epochs=args.save_epoch  # 每个 epoch 保存一次
        )
        best_callback = ModelCheckpoint(
            monitor='train_loss',
            mode='min',
            dirpath=os.path.join(args.save_root, 'checkpoints'),
            filename='best',
            save_top_k=1,
            every_n_epochs=args.save_epoch  # 每个 epoch 保存一次
        )

    trainer = pl.Trainer(
        default_root_dir=args.save_root,
        devices=-1,
        accelerator='gpu' if args.gpu else 'cpu',
        max_epochs=args.epochs,
        callbacks=[epoch_callback, best_callback],
    )
    trainer.fit(model, train_loader, ckpt_path=ckpt_path)

def plot(args):
    models = torch.load(file_path)
    # Sample from full path
    sampled_optim_path = sample_frames(models.optim_path, max_frames=args.n_frames)
    # 返回先验的原始优化权重路径、损失路径、精度路径
    optim_path, loss_path, accu_path = zip(
        *[
            (path["flat_w"], path["loss"], path["accuracy"])
            for path in sampled_optim_path
        ]
    )

    print(f"\n# sampled steps in optimization path: {len(optim_path)}")

    """Dimensionality reduction and Loss Grid"""
    print(f"Dimensionality reduction method specified: {args.reduction_method}")
    dim_reduction = DimReduction(
        params_path=optim_path,
        reduction_method=args.reduction_method,
        custom_directions=args.custom_directions,
        seed=args.seed,
    )
    reduced_dict = dim_reduction.reduce()
    path_2d = reduced_dict["path_2d"]  # 投影路径
    directions = reduced_dict["reduced_dirs"]  # 原始方向
    pcvariances = reduced_dict.get("pcvariances")  # PCA 方差（方向相关）

    loss_grid = LossGrid(
        optim_path=optim_path,
        models=models,
        data=datamodule.dataset.tensors,
        path_2d=path_2d,
        directions=directions,
    )
    # animate_contour(
    #     param_steps=path_2d.tolist(),
    #     loss_steps=loss_path,
    #     acc_steps=accu_path,
    #     loss_grid=loss_grid,
    #     pcvariances=pcvariances,
    #     giffps=giffps,
    #     sampling=sampling,
    #     output_to_file=output_to_file,
    #     filename=output_filename+'_contour',
    # )
    # if static:
    #     static_contour(
    #         param_steps=path_2d.tolist(),
    #         loss_grid=loss_grid,
    #         pcvariances=pcvariances,
    #         giffps=giffps,
    #         sampling=sampling,
    #         output_to_file=output_to_file,
    #         filename=output_filename+'_contour',
    #     )
    # if surface:
    #     animate_surface(
    #         param_steps=path_2d.tolist(),
    #         loss_steps=loss_path,
    #         acc_steps=accu_path,
    #         loss_grid=loss_grid,
    #         giffps=giffps,
    #         sampling=sampling,
    #         output_to_file=output_to_file,
    #         filename=output_filename+'_surface',
    #     )
    #     if static:
    #         static_surface(
    #             param_steps=path_2d.tolist(),
    #             loss_grid=loss_grid,
    #             giffps=giffps,
    #             sampling=sampling,
    #             output_to_file=output_to_file,
    #             filename=output_filename+'_surface',
    #         )
    static_contour(
        param_steps=path_2d.tolist(),
        loss_grid=loss_grid,
        pcvariances=pcvariances,
        giffps=giffps,
        sampling=sampling,
        output_to_file=output_to_file,
    )
    static_surface(
        param_steps=path_2d.tolist(),
        loss_grid=loss_grid,
        giffps=giffps,
        sampling=sampling,
        output_to_file=output_to_file,
    )

    if return_data:
        return list(optim_path), list(loss_path), list(accu_path)