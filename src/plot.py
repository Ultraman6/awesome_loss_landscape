# pylint: disable = no-member, unused-variable
import warnings
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import seaborn


def _static_contour(steps, loss_grid, coords, pcvariances, filename="test.png"):
    _, ax = plt.subplots(figsize=(6, 4))
    coords_x, coords_y = coords
    ax.contourf(coords_x, coords_y, loss_grid, levels=35, alpha=0.9, cmap="YlGnBu")
    w1s = [step[0] for step in steps]
    w2s = [step[1] for step in steps]
    (pathline,) = ax.plot(w1s, w2s, color="r", lw=1)

    ax.set_title("MLP")
    ax.set_xlabel(f"principal component 0, {pcvariances[0]:.1%}")
    ax.set_ylabel(f"principal component 1, {pcvariances[1]:.1%}")
    plt.savefig(filename)
    print(f"{filename} created.")

def set_ax(ax, coords_x, coords_y, loss_grid, title, pcvariances=None):
    ax.contourf(coords_x, coords_y, loss_grid, levels=35, alpha=0.9, cmap="YlGnBu")
    ax.set_title(title)

    xlabel_text = "direction 0"
    ylabel_text = "direction 1"
    if pcvariances is not None:
        xlabel_text = f"principal component 0, {pcvariances[0]:.1%}"
        ylabel_text = f"principal component 1, {pcvariances[1]:.1%}"
    ax.set_xlabel(xlabel_text)
    ax.set_ylabel(ylabel_text)

def set_ax_3d(ax, coords_x, coords_y, loss_grid, title, pcvariances=None):
    ax.contourf(coords_x, coords_y, loss_grid, levels=35, alpha=0.9, cmap="YlGnBu")
    ax.set_title(title)

def _animate_progress(current_frame, total_frames):
    print("\r" + f"Processing {current_frame+1}/{total_frames} frames...", end="")
    if current_frame + 1 == total_frames:
        print("\nConverting to gif, this may take a while...")

def sample_frames(steps, max_frames):
    """Sample uniformly from given list of frames.

    Args:
        steps: The frames to sample from.
        max_frames: Maximum number of frames to sample.

    Returns:
        The list of sampled frames.
    """
    samples = []
    steps_len = len(steps)
    if max_frames > steps_len:
        warnings.warn(
            f"Less than {max_frames} frames provided, producing {steps_len} frames."
        )
        max_frames = steps_len
    interval = steps_len // max_frames
    counter = 0
    for i in range(steps_len - 1, -1, -1):  # Sample from the end
        if i % interval == 0 and counter < max_frames:
            samples.append(steps[i])
            counter += 1

    return list(reversed(samples))

def animate_contour(
    param_steps,
    loss_steps,
    acc_steps,
    loss_grid,
    args
):
    """Draw the frames of the animation.

    Args:
        param_steps: The list of full-dimensional flattened models parameters.
        loss_steps: The list of loss values during training.
        acc_steps: The list of accuracy values during training.
        loss_grid: The origin slice of loss landscape.
    """

    loss_grid_2d = loss_grid.loss_values_log_2d
    true_optim_point = loss_grid.true_optim_point
    true_optim_loss = loss_grid.loss_min
    coords_x, coords_y = loss_grid.coords

    n_frames = len(param_steps)
    print(f"\nTotal frames to process: {n_frames}, result frames per second: {15}")

    fig, ax = plt.subplots(figsize=(9, 6))
    set_ax(ax, coords_x, coords_y, loss_grid_2d, "Optimization in Loss Landscape", True)

    W0 = param_steps[0]
    w1s = [W0[0]]
    w2s = [W0[1]]

    (pathline,) = ax.plot(w1s, w2s, color="r", lw=1)
    (point,) = ax.plot(W0[0], W0[1], "ro")
    (optim_point,) = ax.plot(
        true_optim_point[0], true_optim_point[1], "bx", label="target local minimum"
    )
    plt.legend(loc="upper right")

    step_text = ax.text(
        0.05, 0.9, "", fontsize=10, ha="left", va="center", transform=ax.transAxes
    )
    value_text = ax.text(
        0.05, 0.75, "", fontsize=10, ha="left", va="center", transform=ax.transAxes
    )

    def animate(i): # 绘制优化路径
        W = param_steps[i]
        w1s.append(W[0])
        w2s.append(W[1])
        pathline.set_data(w1s, w2s)
        point.set_data(W[0], W[1])
        step_text.set_text(f"step: {i}")
        value_text.set_text(
            f"loss: {loss_steps[i]: .3f}\nacc: {acc_steps[i]: .3f}\n\n"
            f"target coords: {true_optim_point[0]: .3f}, {true_optim_point[1]: .3f}\n"
            f"target loss: {true_optim_loss: .3f}"
        )

    global anim
    anim = FuncAnimation(
        fig, animate, frames=len(param_steps), interval=100, blit=False, repeat=False
    )

    print(f"Writing {filename}.")
    anim.save(
        f"./{filename}",
        writer="imagemagick",
        fps=15,
        progress_callback=_animate_progress,
    )
    print(f"\n{filename} created successfully.")
    plt.ioff()
    plt.show()
    plt.show()

def static_contour(
    param_steps,
    loss_grid,
    pcvariances,
    giffps,
    sampling=False,
    max_frames=300,
    figsize=(9, 6),
    output_to_file=True,
    filename="static_contour_2d",
):
    """Draw the frames of the animation.

    Args:
        param_steps: The list of full-dimensional flattened models parameters.
        loss_grid: The origin slice of loss landscape.
        coords: The coordinates of the 2D slice.
        true_optim_point: The coordinates of the minimum point in the loss grid.
        pcvariances: Variances explained by the principal components.
        giffps: Frames per second in the output.
        sampling (optional): Whether to sample from the steps. Defaults to False.
        max_frames (optional): Max number of frames to sample. Defaults to 300.
        figsize (optional): Figure size. Defaults to (9, 6).
        output_to_file (optional): Whether to write to file. Defaults to True.
        filename (optional): Defaults to "test.gif".
    """
    loss_grid_2d = loss_grid.loss_values_log_2d
    true_optim_point = loss_grid.true_optim_point
    coords_x, coords_y = loss_grid.coords
    if sampling:
        print(f"\nSampling {max_frames} from {len(param_steps)} input frames.")
        param_steps = sample_frames(param_steps, max_frames)

    n_frames = len(param_steps)
    print(f"\nTotal frames to process: {n_frames}, result frames per second: {giffps}")

    fig, ax = plt.subplots(figsize=figsize)
    set_ax(ax, coords_x, coords_y, loss_grid_2d, "Optimization in Loss Landscape", pcvariances)
    # 静态图模式：绘制完整优化路径-提取所有参数点坐标
    w1s = [W[0] for W in param_steps]
    w2s = [W[1] for W in param_steps]

    # 绘制完整路径线和关键点
    ax.plot(w1s, w2s, color="r", lw=0.1, label="optimization path")
    ax.plot(true_optim_point[0], true_optim_point[1], "bx", label="target minimum")
    ax.scatter(w1s[0], w2s[0], color="r", marker="o", label="start point")
    for i in range(len(param_steps)):
        ax.plot(w1s[i], w2s[i], "ro", markersize=3)
    ax.scatter(w1s[-1], w2s[-1], color="darkred", marker="*", s=100, label="end point")

    # 输出处理
    if output_to_file:
        filename += ".pdf"
        plt.savefig(f"./{filename}", bbox_inches='tight', format='pdf')
        print(f"Static landscape saved as {filename}")
    else:
        plt.show()

def animate_surface(
    param_steps,
    loss_steps,
    acc_steps,
    loss_grid,
    giffps,
    sampling=False,
    max_frames=300,
    figsize=(9, 6),
    output_to_file=True,
    filename="surface",
):
    """Draw the frames of the animation with a 3D surface landscape.

    Args:
        param_steps: The list of full-dimensional flattened models parameters.
        loss_steps: The list of loss values during training.
        acc_steps: The list of accuracy values during training.
        surf_file: The file containing surface datasets.
        surf_name: The name of the surface datasets in the file.
        true_optim_point: The coordinates of the minimum point in the loss grid.
        true_optim_loss: The loss value of the minimum point.
        pcvariances: Variances explained by the principal components.
        giffps: Frames per second in the output.
        sampling (optional): Whether to sample from the steps. Defaults to False.
        max_frames (optional): Max number of frames to sample. Defaults to 300.
        figsize (optional): Figure size. Defaults to (9, 6).
        output_to_file (optional): Whether to write to file. Defaults to True.
        filename (optional): Defaults to "test.gif".
    """
    if sampling:
        print(f"\nSampling {max_frames} from {len(param_steps)} input frames.")
        param_steps = sample_frames(param_steps, max_frames)
        loss_steps = sample_frames(loss_steps, max_frames)
        acc_steps = sample_frames(acc_steps, max_frames)

    n_frames = len(param_steps)
    print(f"\nTotal frames to process: {n_frames}, result frames per second: {giffps}")

    # Load the surface datasets
    true_optim_point = loss_grid.true_optim_point
    true_optim_loss =  loss_grid.loss_min

    x, y = loss_grid.coords
    X, Y = np.meshgrid(x, y)
    Z = loss_grid.loss_values_log_2d
    # Create a 3D landscape using p3 (Axes3D)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)  # Create 3D Axes using p3

    # Plot the surface
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    # Set up initial landscape state
    W0 = param_steps[0]
    w1s = [W0[0]]
    w2s = [W0[1]]
    z_vals = [Z[int(W0[1]), int(W0[0])]]  # Z values for the path

    # Use scatter3D for the initial point
    (point,) = ax.plot([W0[0]], [W0[1]], [z_vals[0]], "ro")
    (optim_point,) = ax.plot(
        true_optim_point[0], true_optim_point[1], true_optim_loss, "bx", label="target local minimum"
    )
    plt.legend(loc="upper right")

    # Use plot3D for the pathline, ensure it's treated as 3D
    pathline, = ax.plot3D([], [], [], color="r", lw=1)

    step_text = ax.text(
        0.05, 0.9, 1.05, s="", fontsize=10, ha="left", va="center", transform=ax.transAxes
    )
    value_text = ax.text(
        0.05, 0.75, 1.05, s="", fontsize=10, ha="left", va="center", transform=ax.transAxes
    )

    def animate(i):  # Update the optimization path
        W = param_steps[i]
        w1s.append(W[0])
        w2s.append(W[1])
        z_vals.append(Z[int(W[1]), int(W[0])])  # Get the Z value at the current W1, W2

        # Update the pathline and its z-values in 3D
        pathline.set_data(w1s, w2s)
        pathline.set_3d_properties(z_vals)

        # Update the current point in 3D
        point.set_data([W[0]], [W[1]])
        point.set_3d_properties([z_vals[-1]])

        # Update the text info
        step_text.set_text(f"step: {i}")
        value_text.set_text(
            f"loss: {loss_steps[i]: .3f}\nacc: {acc_steps[i]: .3f}\n\n"
            f"target coords: {true_optim_point[0]: .3f}, {true_optim_point[1]: .3f}\n"
            f"target loss: {true_optim_loss: .3f}"
        )

    # Call the animator
    global anim
    anim = FuncAnimation(
        fig, animate, frames=len(param_steps), interval=100, blit=False, repeat=False
    )

    if output_to_file:
        filename += ".gif"
        print(f"Writing {filename}.")
        anim.save(
            f"./{filename}",
            writer="imagemagick",
            fps=giffps,
            progress_callback=_animate_progress,
        )
        print(f"\n{filename} created successfully.")
    else:
        plt.ioff()
        plt.show()

def static_surface(
    param_steps,
    loss_grid,
    giffps,
    sampling=False,
    max_frames=300,
    figsize=(9, 6),
    output_to_file=True,
    filename="static_surface_3d",
):
    """Draw a static 3D landscape showing optimization path and loss landscape.

    Args:
        param_steps: The list of full-dimensional flattened models parameters.
        loss_grid: The 2D slice of loss landscape.
        coords: The coordinates of the 2D slice.
        true_optim_point: The coordinates of the minimum point in the loss grid.
        true_optim_loss: The loss value of the minimum point.
        giffps: Frames per second in the output.
        sampling (optional): Whether to sample from the steps. Defaults to False.
        max_frames (optional): Max number of frames to sample. Defaults to 300.
        figsize (optional): Figure size. Defaults to (9, 6).
        output_to_file (optional): Whether to write to file. Defaults to True.
        filename (optional): Defaults to "test_3d.png".
    """
    coords_x, coords_y = loss_grid.coords
    print(len(coords_x))
    true_optim_point = loss_grid.true_optim_point
    true_optim_loss = loss_grid.loss_min
    if sampling:
        print(f"\nSampling {max_frames} from {len(param_steps)} input frames.")
        param_steps = sample_frames(param_steps, max_frames)

    n_frames = len(param_steps)
    print(f"\nTotal frames to process: {n_frames}, result frames per second: {giffps}")

    # 初始化3D画布
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # 生成网格数据（关键修正点）
    X, Y = np.meshgrid(coords_x, coords_y)
    Z = loss_grid.loss_values_log_2d.T  # 转置以匹配网格形状[3,6](@ref)

    # 绘制3D曲面（模仿示例参数）
    surf = ax.plot_surface(X, Y, Z,
                           cmap='rainbow',  # 颜色映射[5,10](@ref)
                           rstride=1,  # 减少行采样密度提升性能[4,6](@ref)
                           cstride=1,
                           edgecolor='none',  # 隐藏网格线[6](@ref)
                           alpha=1.0)  # 调整透明度[4](@ref)

    # 设置坐标轴格式（示例风格）
    ax.zaxis.set_major_locator(LinearLocator(6))  # Z轴刻度定位器[3](@ref)
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))  # 格式化标签[3](@ref)
    ax.set_xlabel('W1', fontsize=12, labelpad=10)  # 标签间距优化[10](@ref)
    ax.set_ylabel('W2', fontsize=12, labelpad=10)
    ax.set_zlabel('Loss', fontsize=12, labelpad=10)

    # 添加颜色条（位置与示例一致）
    # cbar = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.08)
    # cbar.set_label('Loss Value', rotation=270, labelpad=15)  # 颜色条标签[6](@ref)

    # 提取参数路径点（修复索引错误）
    # 路径绘制：1. 路径映射到网格 2. 路径补充到网格（包括坐标与真实损失值）
    w1s = np.array([int(W[1]) for W in param_steps])
    w2s = np.array([int(W[0]) for W in param_steps])
    z_vals = [Z[int(W[0]), int(W[1])] for W in param_steps]

    # 绘制优化路径（增强可视化）
    ax.plot(w1s, w2s, z_vals,
            color='darkorange',
            marker='o', markersize=4,
            linestyle='--', linewidth=1.5,
            alpha=0.8,
            label='Optimization Path')

    # 标注关键点（示例风格优化）
    ax.scatter(true_optim_point[1], true_optim_point[0], true_optim_loss,
               color='lime', marker='X', s=200,
               edgecolor='black', depthshade=False,  # 禁用深度阴影[6](@ref)
               label='Global Minimum')

    ax.scatter(w1s[0], w2s[0], z_vals[0],
               color='red', marker='P', s=150,
               edgecolor='black', zorder=4,
               label='Start Point')

    ax.scatter(w1s[-1], w2s[-1], z_vals[-1],
               color='darkred', marker='*', s=200,
               edgecolor='gold', zorder=4,
               label='End Point')

    # 设置视角与样式（示例推荐参数）
    ax.view_init(elev=25, azim=45)  # 俯仰角与方位角[10](@ref)
    ax.xaxis.pane.fill = False  # 透明坐标平面[6](@ref)
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linestyle=':', alpha=0.6)  # 虚线网格[10](@ref)

    # 图例位置优化（避免遮挡）
    ax.legend(loc='upper right',
              bbox_to_anchor=(1.18, 0.9),  # 偏移图例位置[4](@ref)
              framealpha=0.9)

    # 输出处理
    # if output_to_file:
    #     filename += ".pdf"
    #     fig.colorbar(surf, shrink=0.5, aspect=5)
    #     fig.savefig(f"./{filename}", bbox_inches='tight', format='pdf')
    #     print(f"Static 3D landscape saved as {filename}")
    # else:
    plt.show()

def heat_map(loss_grid, vmin=0.1, vmax=10, filename="heatmap"):
    Z = loss_grid.loss_values_log_2d.T  # 转置以匹配网格形状(y行x列->x行y列)
    sns_plot = seaborn.heatmap(Z, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax,
                           xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    sns_plot.get_figure().savefig(filename + '.pdf',
                                  dpi=300, bbox_inches='tight', format='pdf')

