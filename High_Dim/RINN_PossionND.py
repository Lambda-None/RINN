import sys
sys.path.append(r"\Lib4RINN")

import torch
import time
import numpy as np
import random

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

# 测试代码
if __name__ == "__main__":
    n = 10
    dtype = torch.float64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    domain_1d = torch.tensor((-1, 1), dtype=dtype) 
    domain = domain_1d.repeat(n)  

    N_bcs = 1024
    X_ini, X_bcs = BcsPoint.hypersquare(domain, N_bcs, dtype=dtype)

    N_res = 8192*2
    X_res = ResPoint.hypersquare(domain, N_res, dtype=dtype)

    X_ini = X_ini.to(device)
    X_bcs = X_bcs.to(device)
    X_res = X_res.to(device)
    domain = domain.to(device)

    PDE = PoissonND(domain)
    
    # ######################### PIELM ##############################
    mlp_layers = [n, 3560, 1]
    act = 'tanh'
    w_init = 'uniform_1'

    method = 'PIELM' #'PIELM' or 'RINN' or 'RINN_es'

    pielm = Hard_PIELM(mlp_layers, act, w_init, domain).to(device, dtype)

    start_time = time.time()

    if method == 'PIELM':
        best_coef_solut, _ = PDE.auto_solver(pielm, X_ini, X_bcs, X_res)

    elif method == 'RINN':
        epoch = 1
        trainer = Trainer(pielm, epsilon=0.001)
        trainer.train(X_res, epoch, lr=1e-1, print_interval=100)
        
        best_coef_solut, _ = PDE.auto_solver(pielm, X_ini, X_bcs, X_res)

    elif method == 'RINN_es':
        epoch = 10
        trainer = Trainer_ES(pielm, PDE, X_ini, X_bcs, X_res, epsilon=1e-15)
        best_pde_loss, best_coef_solut = trainer.train_auto(X=X_res, n_epoch=epoch, lr=1e-1, print_interval=10, patience=100)
        best_coef_solut = best_coef_solut.to(device=device)

    pielm.output_model.weight.data.copy_(best_coef_solut.t())

    used_time = time.time()-start_time

    ######################### TEST ##############################
    if n == 1:
        NN_test = Test(domain, PDE, pielm, dtype=dtype, device=device)
        X_eval = NN_test.line()
        u_test, u_true = NN_test.predict(X_eval.to(device))
        L2, Rele_L2, Abs_mean = NN_test.error_compute(u_test, u_true)

        print(f"L2 Error: {L2:.4e}")
        print(f"Relative L2 Error: {Rele_L2:.4e}")
        print(f"Absolute Mean Error: {Abs_mean:.4e}")

        u_test = u_test.detach().cpu().numpy()
        u_true = u_true.detach().cpu().numpy()
        
        visual = visualization(u_test, u_true)

        visual.plot_solutions_1D(X_eval)
    
    elif n == 2:
        NN_test = Test(domain, PDE, pielm, dtype=dtype, device=device)
        X, Y, X_eval = NN_test.rectangle()
        u_test, u_true = NN_test.predict(X_eval.to(device))
        L2, Rele_L2, Abs_mean = NN_test.error_compute(u_test, u_true)

        print(f"L2 Error: {L2:.4e}")
        print(f"Relative L2 Error: {Rele_L2:.4e}")
        print(f"Absolute Mean Error: {Abs_mean:.4e}")

        u_test = u_test.detach().cpu().numpy().reshape(X.shape)
        u_true = u_true.detach().cpu().numpy().reshape(X.shape)

        visual = visualization(u_test, u_true)

        visual.plot_solutions_2D(X, Y)
    
    else:
        L2_list, RelL2_list, AbsMean_list = [], [], []
        seed = torch.randint(10086,12315,(n,))

        for i in range(n):
            torch.manual_seed(seed[i].item())
            NN_test = Test(domain, PDE, pielm, dtype=dtype, device=device)
            X_eval = NN_test.hypersquare(num_eval_pts=10000)
            
            u_test, u_true = NN_test.predict(X_eval.to(device))
            L2, Rele_L2, Abs_mean = NN_test.error_compute(u_test, u_true)

            L2_list.append(L2.item())
            RelL2_list.append(Rele_L2.item())
            AbsMean_list.append(Abs_mean.item())

            print(f"Run {i+1:02d}: L2={L2:.4e}, RelL2={Rele_L2:.4e}, AbsMean={Abs_mean:.4e}")

        L2_mean, L2_std = torch.tensor(L2_list).mean(), torch.tensor(L2_list).std()
        RelL2_mean, RelL2_std = torch.tensor(RelL2_list).mean(), torch.tensor(RelL2_list).std()
        AbsMean_mean, AbsMean_std = torch.tensor(AbsMean_list).mean(), torch.tensor(AbsMean_list).std()

        print("\n==== Summary ====")
        print(f"Mean L2 Error: {L2_mean:.2e} ± {L2_std:.2e}")
        print(f"Mean Relative L2 Error: {RelL2_mean:.2e} ± {RelL2_std:.2e}")
        print(f"Mean Absolute Mean Error: {AbsMean_mean:.2e} ± {AbsMean_std:.2e}")

