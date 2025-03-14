import os.path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.landscape.plot import sample_frames, _animate_progress

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

def animate_contour(
    param_steps,
    loss_steps,
    acc_steps,
    loss_grid,
    args
):
    loss_grid_2d = loss_grid.loss_values_2d
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

    filename = os.path.join(args.plot_root, "animate_contour.gif")
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
    args
):
    loss_grid_2d = loss_grid.loss_values_log_2d
    true_optim_point = loss_grid.true_optim_point
    coords_x, coords_y = loss_grid.coords
    n_frames = len(param_steps)
    print(f"\nTotal frames to process: {n_frames}, result frames per second: {15}")

    fig, ax = plt.subplots(figsize=(9, 6))
    set_ax(ax, coords_x, coords_y, loss_grid_2d, "Optimization in Loss Landscape", True)
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
    filename = os.path.join(args.plot_root, "static_contour.pdf")
    plt.savefig(f"./{filename}", bbox_inches='tight', format='pdf')
    print(f"Static plot saved as {filename}")
    plt.show()
