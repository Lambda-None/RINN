import sys
sys.path.append(r"\Lib4RINN")

import torch
import time
import numpy as np
import random

from Lib4RINN.dataset import *
from Lib4RINN.ELM import *
from Lib4RINN.pde import *
from Lib4RINN.test import *
from Lib4RINN.train import *
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
    type = torch.float64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    domain = torch.tensor((-1, 1), dtype=type)

    num_bcs_point = 1
    X_ini, X_bcs = BcsPoint.line(domain, num_bcs_point, dtype=type)

    num_res_point = 1024
    X_res = ResPoint.line(domain, num_res_point, dtype=type)
    X_res = X_res.to(device)
    X_bcs = X_bcs.to(device)
    domain = domain.to(device)

    PDE = Poisson1D_1(domain)
    
    # ######################### PIELM ##############################
    mlp_layers = [1, 128, 1]
    act = 'tanh'
    w_init = 'uniform_20'

    method = 'RINN_es' #'PIELM' or 'RINN' or 'RINN_es'

    pielm = PIELM(mlp_layers, act, w_init).to(device, type)

    start_time = time.time()

    if method == 'PIELM':
        A, B = PDE.build_linear_system(pielm, X_ini, X_bcs, X_res)
        best_coef_solut = torch.linalg.lstsq(A, B).solution
    elif method == 'RINN':
        epoch = 2000
        trainer = Trainer(pielm, epsilon=0.1)
        trainer.train(X_res, epoch, lr=1e-3, print_interval=100)
        A, B = PDE.build_linear_system(pielm, X_ini, X_bcs, X_res)
        best_coef_solut = torch.linalg.lstsq(A, B).solution
    elif method == 'RINN_es':
        epoch = 2000
        trainer = Trainer_ES(pielm, PDE, X_ini, X_bcs, X_res, epsilon=0.1)
        best_pde_loss, best_coef_solut = trainer.train_auto(X=X_res, n_epoch=epoch, lr=1e-3, print_interval=10, patience=500)

    used_time = time.time()-start_time


    ######################### TEST ##############################
    NN_test = Test(domain, PDE, pielm, dtype=type)
    X_eval = NN_test.line()
    u_test, u_true = NN_test.predict(X_eval.to(device), best_coef_solut)
    L2, Rele_L2, Abs_mean = NN_test.error_compute(u_test, u_true)

    print(f"L2 Error: {L2:.4e}")
    print(f"Relative L2 Error: {Rele_L2:.4e}")
    print(f"Absolute Mean Error: {Abs_mean:.4e}")

    u_test = u_test.detach().cpu().numpy()
    u_true = u_true.detach().cpu().numpy()
    
    visual = visualization(u_test, u_true)

    if method == 'PIELM':
        visual.plot_solutions_1D(X_eval, save_path="png_save/PIELM",filename="PIELM_Possion1D.png")
    
    elif method == 'RINN':
        visual.plot_solutions_1D(X_eval, save_path="png_save/RINN",filename="RINN_Possion1D.png")

    elif method == 'RINN_es':
        visual.plot_solutions_1D(X_eval, save_path="png_save/RINN_es",filename="RINN_es_Possion1D.png")
    else:
        print("Don't need visualizetion")
    

