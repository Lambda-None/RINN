import sys
sys.path.append(r"\Lib4RINN")

import gc
import torch
import random
import numpy as np
from torchmin import least_squares

from Lib4RINN.Dataset import *
from Lib4RINN.Net import *
from Lib4RINN.Pde import *
from Lib4RINN.Test import *
from Lib4RINN.Train import *
from Lib4RINN.Visualization import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(114514)

def sample_with_multiplicative_noise(domain, PDE, N, noise_level, device, dtype):
    domain = torch.as_tensor(domain, device=device, dtype=dtype).reshape(-1, 2)
    mins = domain[:, 0]
    rngs = domain[:, 1] - mins

    X_data = torch.rand(N, domain.shape[0], device=device, dtype=dtype) * rngs + mins
    u_exact = PDE.exact(X_data).reshape(N, -1)

    zeta = 2.0 * torch.rand_like(u_exact) - 1.0
    u_data = u_exact * (1.0 + noise_level * zeta)
    u_data = u_data.squeeze(-1)

    return X_data, u_data


# 测试代码
if __name__ == "__main__":
    dtype = torch.float64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    domain = torch.tensor((0, 1.4, 0, 1.4), dtype=dtype)

    num_bcs_point = 1024
    X_ini, X_bcs = BcsPoint.rectangle(domain, num_bcs_point, dtype=dtype)

    num_res_point = 2048
    X_res = ResPoint.rectangle(domain, num_res_point, dtype=dtype)

    X_ini = X_ini.to(device)
    X_bcs = X_bcs.to(device)
    X_res = X_res.to(device)

    domain = domain.to(device)

    alpha = torch.tensor(1, device=device, dtype=dtype)
    PDE = PoissonInv(domain, alpha)

    noise_levels = [0, 1e-8, 1e-6, 1e-4, 1e-2, 1]

    methods = ['PIELM', 'RINN']

    all_results = {m: [] for m in methods}

    plot_style = {
        'PIELM':  {'color': 'C0', 'marker': 'o', 'linestyle': '-'},
        'RINN':   {'color': 'C1', 'marker': 's', 'linestyle': '-'},
        'RINN_es':{'color': 'C2', 'marker': '^', 'linestyle': '-'},
    }

    mlp_layers = [2, 512, 1024, 1]
    act = 'tanh'
    w_init = 'uniform_1'

    for method in methods:
        print(f"\n===== Running method: {method} =====")

        for noise_level in noise_levels:
            print(f"  -> noise_level = {noise_level:.1e}")
            pielm = PIELM(mlp_layers, act, w_init).to(device, dtype)
            
            X_data, u_data = sample_with_multiplicative_noise(domain, PDE, N=1024, noise_level=noise_level, device=device, dtype=dtype)

            if method == 'PIELM':
                coef_sol = PDE.auto_solver(pielm, X_res, X_bcs, X_data, u_data)
                alpha_sol = coef_sol[0]
                beta_sol = coef_sol[1:]

            elif method == 'RINN':
                epoch = 500
                trainer = Trainer(pielm, epsilon=0.01)
                trainer.train(X_res, epoch, lr=1e-3, print_interval=100)
        
                coef_sol = PDE.auto_solver(pielm, X_res, X_bcs, X_data, u_data)
                alpha_sol = coef_sol[0]
                beta_sol = coef_sol[1:]

            else:
                raise ValueError("No such method")
            
            pielm.output_model.weight.data.copy_(beta_sol.t())

    ######################### TEST ##############################
            NN_test = Test(domain, PDE, pielm, dtype=dtype, device=device)
            X, T, X_eval = NN_test.rectangle()
            u_test, u_true = NN_test.predict(X_eval)
            L2, Rele_L2, Abs_mean = NN_test.error_compute(u_test, u_true)
            Lambda_error = (torch.abs(alpha_sol-alpha)/torch.abs(alpha)).detach().cpu().item()

            all_results[method].append((float(noise_level), float(Rele_L2), float(Lambda_error)))

            del pielm, X_data, u_data, alpha_sol, beta_sol, X, T, NN_test, X_eval, u_test, u_true
            if 'trainer' in locals():
                try:
                    del trainer
                except Exception:
                    pass

            torch.cuda.empty_cache()
            gc.collect()

        torch.cuda.empty_cache()
        gc.collect()

        print(f"\nMethod: {method}")
        for (nl, rel, lam_err) in all_results[method]:
            print(f"  noise={nl:.1e}, Relative_L2_Error={rel:.3e}, Coefficient_Error={lam_err:.3e}")

        save_path = r'Save_Fig\Inverse\inverse_results.txt'

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("Method\tNoise_Level\tRelative_L2_Error\tCoefficient_Error\n")
            for method in methods:
                for (nl, rel, lam_err) in all_results[method]:
                    f.write(f"{method}\t{nl:.16e}\t{rel:.16e}\t{lam_err:.16e}\n")

        print(f"结果已保存到：{save_path}")


    print("\n===== Summary of All Results =====")
    for method in methods:
        print(f"\nMethod: {method}")
        for (nl, rel, lam_err) in all_results[method]:
            print(f"  noise={nl:.1e}, Relative_L2_Error={rel:.3e}, Coefficient_Error={lam_err:.3e}")


    x_pos = np.arange(len(noise_levels)) 
    
    def format_noise_labels(noise_levels):
        labels = []
        for nl in noise_levels:
            if nl == 0:
                labels.append("0")
            else:
                k = int(np.log10(nl))
                labels.append(rf"$10^{{{k}}}$")
        return labels

    noise_labels = format_noise_labels(noise_levels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax = axes[0]
    for method in methods:
        data = all_results[method]
        rel_vals = [d[1] for d in data]
        style = plot_style[method]
        
        ax.plot(x_pos, rel_vals, label=method,
                marker=style['marker'], linestyle=style['linestyle'],
                markeredgewidth=0.8, linewidth=1.5)

    ax.set_yscale('log')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(noise_labels, fontsize=11, rotation=0)  
    ax.set_xlabel('Noise level', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.set_title('Relative L2 Error vs Noise Level', fontsize=13)
    ax.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.legend(loc='upper left', fontsize=10)

    ax = axes[1]
    for method in methods:
        data = all_results[method]
        lam_vals = [d[2] for d in data]
        style = plot_style[method]
        ax.plot(x_pos, lam_vals, label=method,
                marker=style['marker'], linestyle=style['linestyle'],
                markeredgewidth=0.8, linewidth=1.5)

    ax.set_yscale('log')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(noise_labels, fontsize=11, rotation=0)
    ax.set_xlabel('Noise level', fontsize=12)
    ax.set_ylabel('Coefficient Error |Lambda - beta|', fontsize=12)
    ax.set_title('Coefficient Error vs Noise Level', fontsize=13)
    ax.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.legend(loc='upper left', fontsize=10)

    plt.savefig(r'Save_Fig\Inverse\inverse_comparasion.pdf')
    plt.show()
    
