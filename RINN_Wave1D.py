import sys
sys.path.append(r"\Lib4RINN")

import torch
import time
import numpy as np
import random
from scipy.io import loadmat

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
    domain = torch.tensor((0, 1, 0, 1), dtype=type)
 
    num_ini_point = 1024
    num_bcs_point = 1024
    X_ini, X_bcs = IniBcsPoint.rectangle(domain, num_ini_point, num_bcs_point, dtype=type)

    num_res_point = 2048
    X_res = ResPoint.rectangle(domain, num_res_point, dtype=type)

    X_ini = X_ini.to(device)
    X_bcs = X_bcs.to(device)
    X_res = X_res.to(device)

    domain = domain.to(device)
    c = 2
    PDE = Wave_1(domain, c)
    
    # ######################### PIELM ##############################
    mlp_layers = [2, 512, 1024, 1]
    act = 'tanh'
    w_init = 'uniform_1'
    method = 'RINN_es' # 'PIELM' or 'RINN' or 'RINN_es'

    pielm = PIELM(mlp_layers, act, w_init).to(device, type)
    start_time = time.time()

    if method == 'PIELM':
        A, B = PDE.build_linear_system(pielm, X_ini, X_bcs, X_res)
        best_coef_solut = torch.linalg.lstsq(A, B).solution

    elif method == 'RINN':
        epoch = 500
        trainer = Trainer(pielm, epsilon=0.01)
        trainer.train(X_res, epoch, lr=1e-3, print_interval=10)
        A, B = PDE.build_linear_system(pielm, X_ini, X_bcs, X_res)
        best_coef_solut = torch.linalg.lstsq(A, B).solution

    elif method == 'RINN_es':
        epoch = 1000
        trainer = Trainer_ES(pielm, PDE, X_ini, X_bcs, X_res, epsilon=0.01)
        best_pde_loss, best_coef_solut = trainer.train_auto(X=X_res, n_epoch=epoch, lr=1e-3, print_interval=10, patience=250)

    else:
        raise ValueError("No such method")

    used_time = time.time()-start_time
    print(f'used_time:{used_time:.1f}')
    ######################### TEST ##############################
    NN_test = Test(domain, PDE, pielm, dtype=type)

    data = loadmat(r'evolution_pde_exact\wave1_True.mat')
    t = data['t'].flatten()[:,None] 
    x = data['x'].flatten()[:,None] 
    u = data['u']

    xx, tt = np.meshgrid(x, t)
    X_eval = np.vstack((xx.ravel(), tt.ravel())).T    
    u_true  = u.ravel()[:,None]    
    
    X_eval = torch.from_numpy(X_eval).double().to(device)
    u_true = torch.from_numpy(u_true).double().to(device)

    u_test, _ = NN_test.predict(X_eval.to(device), best_coef_solut)
    L2, Rele_L2, Abs_mean = NN_test.error_compute(u_test, u_true)

    print(f"L2 Error: {L2:.4e}")
    print(f"Relative L2 Error: {Rele_L2:.4e}")
    print(f"Absolute Mean Error: {Abs_mean:.4e}")

    u_test = u_test.detach().cpu().numpy().reshape(xx.shape)
    u_true = u_true.detach().cpu().numpy().reshape(xx.shape)

    visual = visualization(u_test.T, u_true.T)

    if method == 'PIELM':
        visual.plot_solutions_2D(xx, tt, save_path="png_save/PIELM",filename="PIELM_Wave1D.png")
    
    elif method == 'RINN':
        visual.plot_solutions_2D(xx, tt, save_path="png_save/RINN",filename="RINN_Wave1D.png")

    elif method == 'RINN_es':
        visual.plot_solutions_2D(xx, tt, save_path="png_save/RINN_es",filename="RINN_es_Wave1D.png")
    else:
        print("Don't need visualizetion")
