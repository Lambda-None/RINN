import time
import torch
import copy

class Trainer:
    def __init__(self, model, epsilon=0.1):
        self.model = model
        self.epsilon = epsilon

    def independence_loss(self, activations):
        batch_size, num_neurons = activations.shape
        cov = torch.mm(activations.T, activations) / (batch_size - 1)

        diag_mask = torch.eye(num_neurons, dtype=torch.bool, device=activations.device)
        off_diag_mask = ~torch.eye(num_neurons, dtype=torch.bool, device=activations.device)
    
        loss_diag = torch.sum(torch.abs(torch.log10(torch.square(cov[diag_mask]))))
        loss_off_diag = torch.norm(cov[off_diag_mask], p='fro')

        total_loss = self.epsilon * loss_diag + loss_off_diag
        return total_loss, loss_diag, loss_off_diag
    
    def train(self, X, n_epoch, lr=1e-3, print_interval=100):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = []
        self.loss_log_diag = [] 
        self.loss_log_off_diag = [] 
        
        start_time = time.time()      

        print(f"{'Epoch':>8} | {'Time (s)':>9} | {'Total Loss':>12} | {'Diag Loss':>12} | {'Off-Diag Loss':>12}")
        print("-" * 66)  

        for epoch in range(n_epoch):
            optimizer.zero_grad()
            
            H = self.model.get_last_hidden_layer(X)
            
            loss, loss_diag, loss_off_diag = self.independence_loss(H)
            
            loss.backward()
            optimizer.step()
                
            self.loss_log_off_diag.append(loss.item())

            if (epoch + 1) % print_interval == 0:
                elapsed = time.time() - start_time
                print(f"{epoch+1:8d} | {elapsed:9.2f} | {loss.item():12.4e} | {loss_diag.item():12.4e} | {loss_off_diag.item():12.4e}")


class Trainer_ES:
    def __init__(self, model, pde, X_ini, X_bcs, X_res, epsilon=0.1):
        self.model = model
        self.pde = pde
        self.X_ini = X_ini
        self.X_bcs = X_bcs
        self.X_res = X_res
        self.epsilon = epsilon

        self.loss_log = []
        self.loss_pde_log = []

    def independence_loss(self, activations):
        batch_size, num_neurons = activations.shape
        cov = torch.mm(activations.T, activations) / (batch_size - 1)

        diag_mask = torch.eye(num_neurons, dtype=torch.bool, device=activations.device)
        off_diag_mask = ~torch.eye(num_neurons, dtype=torch.bool, device=activations.device)
    
        loss_diag = torch.sum(torch.abs(torch.log10(torch.square(cov[diag_mask]))))
        loss_off_diag = torch.norm(cov[off_diag_mask], p='fro')

        total_loss = self.epsilon * loss_diag + loss_off_diag
        return total_loss, loss_diag, loss_off_diag
    
    def train_auto(self, X, n_epoch, lr=1e-3, print_interval=100, patience=20, update_interval=1):
        for param in self.model.output_model.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(list(self.model.hidden_model.parameters()), lr=lr)

        best_model_state = None
        tol = float('inf')
        best_pde_loss = float('inf')
        best_epoch = -1
        bad_epochs = 0
        self.loss_pde_log = []

        start_time = time.time()

        last_coef_solut, last_pde_loss = self.pde.auto_solver(self.model, self.X_ini, self.X_bcs, self.X_res)

        header = (
            f"{'Epoch':>5} | {'Time(s)':>8} | "
            f"{'TotIndep':>10} | {'Diag':>8} | {'OffDiag':>8} | "
            f"{'PDE':>10} | {'Tol':>8} | {'Update':>6}"
        )
        print(header)
        print('-' * len(header))

        for epoch in range(n_epoch):
            optimizer.zero_grad()
            H = self.model.get_last_hidden_layer(X)
            total_loss, loss_diag, loss_off = self.independence_loss(H)
            total_loss.backward()
            optimizer.step()
            
            should_update = (epoch == 0) or ((epoch + 1) % update_interval == 0) or (epoch == n_epoch - 1)
            
            if should_update:
                coef_solut, pde_loss = self.pde.auto_solver(self.model, self.X_ini, self.X_bcs, self.X_res, last_coef_solut)
                self.model.output_model.weight.data.copy_(coef_solut.t())
                self.loss_pde_log.append(pde_loss.item())
                last_coef_solut = coef_solut
                last_pde_loss = pde_loss
                update_flag = "✓" 
            else:
                pde_loss = last_pde_loss
                update_flag = ""    

            if pde_loss < tol:
                tol = pde_loss
                best_pde_loss = pde_loss
                bad_epochs = 0
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(self.model.state_dict())
                self.best_coef_solut = last_coef_solut.detach().clone() if last_coef_solut is not None else None
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stopping at epoch {epoch+1}: "
                        f"no improvement below tol={tol:.2e} for {patience} epochs.")
                    break
                
            if (epoch + 1) % print_interval == 0 or epoch == 0 or bad_epochs == 0:
                elapsed = time.time() - start_time
                row = (
                    f"{epoch+1:5d} | "
                    f"{elapsed:8.1f} | "
                    f"{total_loss.item():10.2e} | "
                    f"{loss_diag.item():8.2e} | "
                    f"{loss_off.item():8.2e} | "
                    f"{pde_loss.item():10.2e} | "
                    f"{tol:8.2e} | "
                    f"{update_flag:>6}"
                )
                print(row)
                
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Best model restored from epoch {best_epoch} with pde_loss = {best_pde_loss:.2e}.")
            if self.best_coef_solut is not None:
                print(f"Best coef_solut shape: {tuple(self.best_coef_solut.shape)} stored in attribute 'best_coef_solut'.")
            else:
                print("Warning: No valid coefficient solution was stored.")
        
        return best_pde_loss, self.best_coef_solut
        
