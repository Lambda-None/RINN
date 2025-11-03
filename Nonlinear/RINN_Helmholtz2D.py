import sys
sys.path.append(r"\Lib4RINN")

import time
import torch
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
    dtype = torch.float64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    domain = torch.tensor((-1, 1, -1, 1), dtype=dtype)

    num_bcs_point = 1024
    X_ini, X_bcs = BcsPoint.rectangle(domain, num_bcs_point, dtype=dtype)

    num_res_point = 2048
    X_res = ResPoint.rectangle(domain, num_res_point, dtype=dtype)

    X_res = X_res.to(device)
    X_bcs = X_bcs.to(device)
    
    domain = domain.to(device)

    PDE = Helmholtz_2D(domain)
    
    # ######################### PIELM ##############################
    mlp_layers = [2, 512, 1024, 1]
    act = 'tanh'
    w_init = 'uniform_1'

    method = 'PIELM' #'PIELM' or 'RINN' or 'RINN_es'

    pielm = PIELM(mlp_layers, act, w_init).to(device, dtype)

    start_time = time.time()

    if method == 'PIELM':
        best_coef_solut, loss = PDE.auto_solver(pielm, X_ini, X_bcs, X_res)

    elif method == 'RINN':
        epoch = 200
        trainer = Trainer(pielm, epsilon=0.01)
        trainer.train(X_res, epoch, lr=1e-3, print_interval=100)
        
        best_coef_solut, loss = PDE.auto_solver(pielm, X_ini, X_bcs, X_res)

    elif method == 'RINN_es':
        epoch = 200
        trainer = Trainer_ES(pielm, PDE, X_ini, X_bcs, X_res, epsilon=0.01)
        best_pde_loss, best_coef_solut = trainer.train_auto(
            X=X_res, n_epoch=epoch, lr=1e-3, print_interval=10, patience=100, update_interval=10)

    pielm.output_model.weight.data.copy_(best_coef_solut.t())

    used_time = time.time()-start_time
    print(f'Runtime:{used_time:.2f}')

    ######################### TEST ##############################
    NN_test = Test(domain, PDE, pielm, dtype=dtype, device=device)
    X, Y, X_eval = NN_test.rectangle()
    u_test, u_true = NN_test.predict(X_eval.to(device))
    L2, Rele_L2, Abs_mean = NN_test.error_compute(u_test, u_true)
    print(f"L2 Error: {L2:.4e}")
    print(f"Relative L2 Error: {Rele_L2:.2e}")
    print(f"Absolute Mean Error: {Abs_mean:.4e}")

    X_plot = X.cpu()
    Y_plot = Y.cpu()

    u_test = u_test.detach().cpu().numpy().reshape(X.shape)
    u_true = u_true.detach().cpu().numpy().reshape(X.shape)

    visual = visualization(u_test, u_true)

    if method == 'PIELM':
        visual.plot_solutions_2D(X_plot, Y_plot, save_path="Save_Fig/noneLinear",filename="PIELM_Helmholtz2D.pdf")
    
    elif method == 'RINN':
        visual.plot_solutions_2D(X_plot, Y_plot, save_path="Save_Fig/noneLinear",filename="RINN_Helmholtz2D.pdf")

    elif method == 'RINN_es':
        visual.plot_solutions_2D(X_plot, Y_plot, save_path="Save_Fig/noneLinear",filename="RINN_es_Helmholtz2D.pdf")
    else:
        print("Don't need visualizetion")

