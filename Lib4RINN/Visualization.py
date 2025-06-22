import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator, FixedFormatter, MaxNLocator

class visualization:
    def __init__(self, u_test, u_true, k=None):
        self.u_test = u_test
        self.u_true = u_true
        self.error = np.abs(u_test - u_true)
        self.k = k

    def plot_solutions_1D(self, X_eval,
                      save_path=None, filename="output.png",
                      axis_num_ticks=5, show_axes=True, show_title=True):
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        label_font = {'fontsize': 14, 'fontweight': 'bold', 'family': 'serif'}
        tick_font = {'labelsize': 12, 'direction': 'in', 'width': 1.5}

        # --------- 子图1：真解 vs 预测解 ---------
        axs[0].plot(X_eval, self.u_true, '-', label='Exact', color='red')
        axs[0].plot(X_eval, self.u_test, '--', label='Pred', color='blue')

        if show_axes:
            axs[0].set_xlabel(r'$x$', fontsize=14, fontweight='bold', family='serif')
            axs[0].set_ylabel(r'$u(x)$', fontsize=14, fontweight='bold', family='serif')
            axs[0].tick_params(labelsize=12, direction='in', width=1.5)
            axs[0].xaxis.set_major_locator(MaxNLocator(nbins=axis_num_ticks, prune=None))
            axs[0].yaxis.set_major_locator(MaxNLocator(nbins=axis_num_ticks, prune=None))
        else:
            axs[0].set_xticks([])
            axs[0].set_yticks([])

        if show_title:
            axs[0].set_title("Exact & Numerical", **label_font)

        axs[0].legend(loc='upper right')
        axs[0].grid(True)

        # --------- 子图2：误差图（含科学计数法刻度） ---------
        axs[1].plot(X_eval, self.error, label='Error', color='green')
        axs[1].legend(loc='upper right')
        axs[1].grid(True)

        if show_axes:
            axs[1].set_xlabel(r'$x$', fontsize=14, fontweight='bold', family='serif')
            axs[1].set_ylabel(r'$Error$', fontsize=14, fontweight='bold', family='serif')
            axs[1].tick_params(labelsize=12, direction='in', width=1.5)

            # ---------- 科学计数法纵坐标 ----------
            num_ticks = axis_num_ticks
            data_max = np.max(self.error)
            exp = int(np.floor(np.log10(data_max))) if data_max > 0 else 0
            tick_vals = np.linspace(0, data_max, num_ticks)

            def format_mantissa(ticks, exp):
                return [f"{(t / (10**exp)):.1f}" if t != 0 else "0.0" for t in ticks]

            mantissa_labels = format_mantissa(tick_vals, exp)

            axs[1].yaxis.set_major_locator(FixedLocator(tick_vals))
            axs[1].set_yticklabels(mantissa_labels)

            axs[1].text(-0.05, 1.0,
                        rf"$\times 10^{{{exp}}}$",
                        transform=axs[1].transAxes,
                        fontsize=10,
                        ha='left', va='bottom')

            axs[1].xaxis.set_major_locator(MaxNLocator(nbins=axis_num_ticks, prune=None))
        else:
            axs[1].set_xticks([])
            axs[1].set_yticks([])

        if show_title:
            axs[1].set_title("Absolute Error", fontsize=14, fontweight='bold', family='serif')

        # ---------- 设置科学计数法 y 轴刻度 ----------
        num_ticks = 5
        data_max = np.max(self.error)
        exp = int(np.floor(np.log10(data_max))) if data_max > 0 else 0
        tick_vals = np.linspace(0, data_max, num_ticks)

        def format_mantissa(ticks, exp):
            return [f"{(t / (10**exp)):.1f}" if t != 0 else "0.0" for t in ticks]

        mantissa_labels = format_mantissa(tick_vals, exp)

        axs[1].set_yticks(tick_vals)
        axs[1].set_yticklabels(mantissa_labels)

        axs[1].text(-0.05, 1.0,
                    rf"$\times 10^{{{exp}}}$",
                    transform=axs[1].transAxes,
                    fontsize=10,
                    ha='left', va='bottom')

        # ---------- 保存与显示 ----------
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fname = filename if self.k is None else filename.replace('k', f'k={self.k}')
            plt.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches='tight')
        plt.show()


    def _plot_with_colorbar(self, X, Y, ax, data, title,
                        num_ticks=5, zero_min=False,
                        axis_num_ticks=5,
                        show_axes=True, show_title=True,
                        use_imshow=False, cmap='jet',
                        xlabel='x', ylabel='y'):

        vmin = 0.0 if zero_min else np.nanmin(data)
        vmax = np.nanmax(data)
        if vmax == vmin:
            vmax = vmin + 1e-10

        exp = int(np.floor(np.log10(max(abs(vmin), abs(vmax))))) if max(abs(vmin), abs(vmax)) > 0 else 0
        tick_vals = np.linspace(vmin, vmax, num_ticks)
        mantissa_labels = [
            "0.0" if t == 0 else f"{(t / (10**exp)):.1f}"
            for t in tick_vals
        ]

        norm = Normalize(vmin=vmin, vmax=vmax)
        if use_imshow:
            extent = [X.min(), X.max(), Y.min(), Y.max()]
            im = ax.imshow(np.where(np.isnan(data), np.nan, data).T,
                           extent=extent,
                           origin='lower',
                           aspect='auto',
                           cmap=cmap,
                           norm=norm)
        else:
            im = ax.pcolor(X, Y, data, shading='auto', cmap=cmap, norm=norm)

        cbar = plt.colorbar(im, ax=ax, ticks=tick_vals)
        cbar.ax.yaxis.set_major_locator(FixedLocator(tick_vals))
        cbar.ax.set_yticklabels(mantissa_labels)
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.text(-0.5, 1.0,
                     rf"$\times 10^{{{exp}}}$",
                     transform=cbar.ax.transAxes,
                     fontsize=12, ha='left', va='bottom')

        if show_axes:
            ax.set_xlabel(rf'${xlabel}$', fontsize=14, fontweight='bold', family='serif')
            ax.set_ylabel(rf'${ylabel}$', fontsize=14, fontweight='bold', family='serif')
            ax.tick_params(labelsize=14, direction='in', width=1.5)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=axis_num_ticks))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=axis_num_ticks))
        else:
            ax.set_xticks([]); ax.set_yticks([])

        if show_title:
            ax.set_title(title, fontsize=14, fontweight='bold', family='serif')

        return im


    def plot_solutions_2D(self, X, Y, save_path=None, filename="output.png",
                      axis_num_ticks=5, show_axes=True, show_title=True,
                      use_imshow=True,
                      xlabel='x', ylabel='y'):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        for ax, data, title in zip(
            axs[:2], [self.u_true, self.u_test],
            ['Exact Solution', 'Numerical Solution']
        ):
            self._plot_with_colorbar(
                X, Y, ax, data, title,
                num_ticks=5, zero_min=False,
                axis_num_ticks=axis_num_ticks,
                show_axes=show_axes, show_title=show_title,
                use_imshow=use_imshow,
                xlabel=xlabel, ylabel=ylabel  # 明确指定坐标轴标签
            )

        self._plot_with_colorbar(
            X, Y, axs[2], self.error, 'Absolute Error',
            num_ticks=5, zero_min=True,
            axis_num_ticks=axis_num_ticks,
            show_axes=show_axes, show_title=show_title,
            use_imshow=use_imshow,
            xlabel=xlabel, ylabel=ylabel
        )

        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fname = filename if self.k is None else filename.replace('k', f'k={self.k}')
            plt.savefig(os.path.join(save_path, fname), dpi=150, bbox_inches='tight')
        plt.show()

    def plot_solutions_Lshape(self, X, Y, save_path=None, filename="output.png",
                              axis_num_ticks=5, show_axes=True, show_title=True,
                              use_imshow=True,
                              xlabel='x', ylabel='y'):
        # 1. 计算 L 型掩码（与训练域保持一致）
        xmin, xmax = X.min(), X.max()
        ymin, ymax = Y.min(), Y.max()
        x_mid = (xmin + xmax) / 2
        y_mid = (ymin + ymax) / 2

        # L 型区域定义：x <= x_mid 或 y >= y_mid
        mask = (X <= x_mid) | (Y >= y_mid)

        # 2. 对三组数据应用掩码，掩码外设为 nan
        u_true_mask = np.where(mask, self.u_true, np.nan)
        u_test_mask = np.where(mask, self.u_test, np.nan)
        err_mask    = np.where(mask, self.error,   np.nan)

        # 3. 创建三列子图
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        # Exact 解
        self._plot_with_colorbar(
            X, Y, axs[0], u_true_mask, "Exact Solution (L-shape)",
            num_ticks=5, zero_min=False,
            axis_num_ticks=axis_num_ticks,
            show_axes=show_axes, show_title=show_title,
            use_imshow=use_imshow,
            xlabel=xlabel, ylabel=ylabel
        )

        # 数值解
        self._plot_with_colorbar(
            X, Y, axs[1], u_test_mask, "Numerical Solution (L-shape)",
            num_ticks=5, zero_min=False,
            axis_num_ticks=axis_num_ticks,
            show_axes=show_axes, show_title=show_title,
            use_imshow=use_imshow,
            xlabel=xlabel, ylabel=ylabel
        )

        # 绝对误差
        self._plot_with_colorbar(
            X, Y, axs[2], err_mask, "Absolute Error (L-shape)",
            num_ticks=5, zero_min=True,
            axis_num_ticks=axis_num_ticks,
            show_axes=show_axes, show_title=show_title,
            use_imshow=use_imshow,
            xlabel=xlabel, ylabel=ylabel
        )

        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fname = filename if self.k is None else filename.replace('k', f'k={self.k}')
            plt.savefig(os.path.join(save_path, fname), dpi=150, bbox_inches='tight')
        plt.show()
